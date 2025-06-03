mod args;
mod bigram_language_model;
mod device;
mod language_model;
mod tokenizer;
mod transformer_language_model;
mod util;

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use args::Args;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use language_model::{language_generate_and_print, language_loss};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokenizer::Tokenizer;
use util::{count_params, print_gradient_info};

/// After how many epochs do we evaluate the model again?
const EVAL_INTERVAL: usize = 500;

/// How many batches to compute loss over.
const EVAL_ITERS: usize = 200;

/// This is based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out.":
///
///     https://youtu.be/kCc8FmEb1nY
fn main() -> Result<()> {
    let multi_progress = MultiProgress::new();
    let args = Args::parse();
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let seed = args.seed.unwrap_or(timestamp);
    let mut rng = StdRng::seed_from_u64(seed);
    let device = args.device.to_candle_device()?;
    println!("Using {} for training/inference.", args.device);

    let training_corpus = std::fs::read_to_string(&args.corpus)?;
    let tokenizer = Tokenizer::from_string(&training_corpus)?;
    let vocab_size = tokenizer.len();
    println!("Initialized tokenizer with {} tokens.", vocab_size);

    let data = Tensor::new(tokenizer.encode(&training_corpus)?, &device)?;
    let data_len = data.shape().dim(0)?;

    println!("Data is {} tokens.", data_len);
    let train_split_index = (0.9 * (data_len as f64)) as usize;
    let train_data = data.i(..train_split_index)?;
    let val_data = data.i(train_split_index..)?;
    println!("Training data is {} tokens.", train_data.shape().dim(0)?);
    println!("Validation data is {} tokens.", val_data.shape().dim(0)?);

    let get_batch = |data: &Tensor, rng: &mut StdRng| -> Result<(Tensor, Tensor)> {
        let mut x = Vec::with_capacity(args.batch_size);
        let mut y = Vec::with_capacity(args.batch_size);
        let data_len = data.shape().dim(0)?;
        for _ in 0..args.batch_size {
            // Ideally we'd use candle for these random numbers, but as far as I can tell,
            // it can only generate random floats. I guess we could round/cast them to
            // integers but for now I'm just going to use the rand crate instead.
            //
            // Also, I think this might be doing a lot of back-and-forth between GPU and CPU.
            // If we can use Candle for all of this, and only use Tensors, rather than pulling
            // from the data Tensor into a Vec and then converting it back into a Tensor,
            // this might be able to run faster on GPUs.
            let idx: usize = rng.random_range(0..(data_len - args.block_size));
            x.push(data.i(idx..(args.block_size + idx))?);
            y.push(data.i((idx + 1)..(args.block_size + idx + 1))?);
        }
        Ok((Tensor::stack(&x, 0)?, Tensor::stack(&y, 0)?))
    };

    // let (xs, ys) = get_batch(&train_data, &mut rng)?;
    // println!("xs:\n{xs}\nys:\n{ys}");

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = args.create_model(vocab_size, vb)?;

    if let Some(load) = &args.load {
        let load = normalize_safetensors_filename(load);
        println!("Loading weights from {load}.");
        varmap.load(load)?;
    }

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )?;

    println!("Parameters in model: {}", count_params(&varmap));

    if args.vars {
        let data = varmap.data().lock().unwrap();
        println!("varmap vars: {:#?}", data);
    }

    let estimate_loss = |name: &str, data: &Tensor, rng: &mut StdRng| -> Result<f32> {
        let mut losses = Vec::with_capacity(EVAL_ITERS);
        let model_no_grad = args.create_model_no_grad(vocab_size, &varmap, &device)?;
        let bar = multi_progress.add(ProgressBar::new(EVAL_ITERS as u64));
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:30.white/white} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        bar.set_message(format!("Estimating {name}"));
        for _ in 0..EVAL_ITERS {
            // I asked on HF Discord and someone said the analog of torch.no_grad
            // is calling .detach() on inputs, but I'm not really sure if the advice I was
            // given is correct. Try adding the following logging just after we call get_batch():
            //
            //     println!("xs is_variable={} track_op={}", xs.is_variable(), xs.track_op());
            //
            // This will show false for both, which makes sense--we don't _want_ to treat
            // the input as a variable or track its gradient. Then looking at the implementation
            // for is_variable(), track_op() and detach(), it's clear that `xs` is just being
            // cloned when we call detach() and nothing more... If that's the case, it means our
            // gradients might get totally messed up every time we estimate the loss.
            //
            // Another way to demonstrate this is by actually using detach() during _training_
            // and displaying the gradients. If detach() is working, then no gradients should
            // actually be calculated, but running the CLI with `--vars` indicates that they
            // _are_ being calculated.
            //
            // I think the "real" way to disable backprop calculation is to actually detach
            // the _variables_, not the inputs, which is why I wrote `create_model_no_grad`.
            let (xs, ys) = get_batch(&data, rng)?;
            let logits = model_no_grad.forward(&xs)?;
            let loss = language_loss(&logits, &ys)?;
            losses.push(loss.to_scalar()?);
            bar.inc(1);
        }
        let loss = losses.iter().sum::<f32>() / losses.len() as f32;
        bar.finish_and_clear();
        Ok(loss)
    };

    let calculate_loss = |prefix: String, rng: &mut StdRng| -> Result<()> {
        let train_loss = estimate_loss("train loss", &train_data, rng)?;
        let val_loss = estimate_loss("val loss", &val_data, rng)?;
        println!("{prefix} train loss {train_loss:.4}, val loss {val_loss:.4}",);
        Ok(())
    };

    let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let main_pb = multi_progress.add(ProgressBar::new(args.epochs as u64));
    main_pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    if args.epochs > 0 {
        calculate_loss("Initial".to_owned(), &mut rng)?;
    }
    for i in 1..=args.epochs {
        main_pb.set_message("Training");
        let (xs, ys) = get_batch(&train_data, &mut rng)?;
        let logits = model.forward(&xs)?;
        let loss = language_loss(&logits, &ys)?;
        let gradients = loss.backward()?;
        if args.vars && i == args.epochs {
            print_gradient_info(&varmap, &gradients)?;
        }
        optimizer.step(&gradients)?;
        main_pb.inc(1);

        if i % EVAL_INTERVAL == 0 || i == args.epochs {
            calculate_loss(format!("Epoch {i}"), &mut rng)?;
        }
    }

    if args.epochs > 0 {
        let end_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        println!("Total training time: {} ms", end_time - start_time)
    }

    if let Some(save) = &args.save {
        let save = normalize_safetensors_filename(save);
        println!("Saving weights to {save}.");
        varmap.save(save)?;
    }

    if args.chars > 0 {
        let mut rng = StdRng::seed_from_u64(seed);
        let model_no_grad = args.create_model_no_grad(vocab_size, &varmap, &device)?;
        language_generate_and_print(
            &model_no_grad,
            args.block_size,
            args.chars,
            &mut rng,
            &device,
            &tokenizer,
        )?;
    }

    Ok(())
}

fn normalize_safetensors_filename(filename: &String) -> String {
    add_extension_if_missing(filename, "safetensors")
}

fn add_extension_if_missing(filename: &String, extension: &str) -> String {
    let path = Path::new(filename);
    if path.extension().is_none() {
        let mut new_path = PathBuf::from(path);
        new_path.set_extension(extension);
        new_path.to_string_lossy().into_owned()
    } else {
        filename.to_string()
    }
}
