mod bigram_language_model;
mod language_model;
mod tokenizer;
mod transformer_language_model;
mod util;

use std::{
    collections::HashMap,
    ops::Deref,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use bigram_language_model::BigramLanguageModel;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use language_model::{language_generate, language_loss};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokenizer::Tokenizer;
use transformer_language_model::TransformerLanguageModel;

/// Number of examples in each batch.
const BATCH_SIZE: usize = 32;

/// Context size, in tokens.
const BLOCK_SIZE: usize = 8;

/// After how many epochs do we evaluate the model again?
const EVAL_INTERVAL: usize = 300;

/// How many batches to compute loss over.
const EVAL_ITERS: usize = 200;

#[derive(Debug, Clone, ValueEnum)]
pub enum Model {
    Bigram,
    Transformer,
}

#[derive(Parser)]
pub struct Args {
    /// Whether to display information about the variables in the network.
    #[arg(long, default_value_t = false)]
    pub vars: bool,

    /// Random number seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of epochs to train the model.
    #[arg(long, default_value_t = 3_000)]
    pub epochs: usize,

    /// Number of characters of output to generate.
    #[arg(long, default_value_t = 500)]
    pub chars: usize,

    /// The file to save the trained weights to, in safetensors format.
    #[arg(long)]
    pub save: Option<String>,

    /// The file to load the trained weights from, in safetensors format.
    #[arg(long)]
    pub load: Option<String>,

    #[arg(long, value_enum, default_value_t = Model::Bigram)]
    pub model: Model,

    /// Number of self-attention/feed-forward blocks (used only when model is transformer).
    #[arg(long, default_value_t = 1)]
    pub blocks: usize,

    /// The learning rate.
    #[arg(long, default_value_t = 0.01)]
    pub lr: f64,
}

fn get_tiny_shakespeare() -> Result<String> {
    // https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    let content = std::fs::read_to_string("tiny-shakespeare.txt")?;
    Ok(content)
}

/// This is based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out.":
///
///     https://youtu.be/kCc8FmEb1nY
fn main() -> Result<()> {
    let args = Args::parse();
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let seed = args.seed.unwrap_or(timestamp);
    let mut rng = StdRng::seed_from_u64(seed);
    let device = Device::Cpu;

    let tiny_shakespeare = get_tiny_shakespeare()?;
    let tokenizer = Tokenizer::from_string(&tiny_shakespeare)?;
    let vocab_size = tokenizer.len();
    println!("Initialized tokenizer with {} tokens.", vocab_size);

    if cfg!(debug_assertions) {
        println!("encoded 'hii there': {:?}", tokenizer.encode("hii there")?);
        println!(
            "decoded 'hii there': {:?}",
            tokenizer.decode(&tokenizer.encode("hii there")?)?
        );
    }

    let data = Tensor::new(tokenizer.encode(&tiny_shakespeare)?, &device)?;
    let data_len = data.shape().dim(0)?;

    println!("Data is {} tokens.", data_len);
    let train_split_index = (0.9 * (data_len as f64)) as usize;
    let train_data = data.i(..train_split_index)?;
    let val_data = data.i(train_split_index..)?;
    println!("Training data is {} tokens.", train_data.shape().dim(0)?);
    println!("Validation data is {} tokens.", val_data.shape().dim(0)?);

    let get_batch = |data: &Tensor, rng: &mut StdRng| -> Result<(Tensor, Tensor)> {
        let mut x = Vec::with_capacity(BATCH_SIZE);
        let mut y = Vec::with_capacity(BATCH_SIZE);
        let data_len = data.shape().dim(0)?;
        for _ in 0..BATCH_SIZE {
            // Ideally we'd use candle for these random numbers, but as far as I can tell,
            // it can only generate random floats. I guess we could round/cast them to
            // integers but for now I'm just going to use the rand crate instead.
            let idx: usize = rng.random_range(0..(data_len - BLOCK_SIZE));
            x.push(data.i(idx..(BLOCK_SIZE + idx))?);
            y.push(data.i((idx + 1)..(BLOCK_SIZE + idx + 1))?);
        }
        Ok((Tensor::stack(&x, 0)?, Tensor::stack(&y, 0)?))
    };

    // let (xs, ys) = get_batch(&train_data, &mut rng)?;
    // println!("xs:\n{xs}\nys:\n{ys}");

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let create_model = |vb: VarBuilder| -> Result<Box<dyn Module>> {
        match args.model {
            Model::Bigram => Ok(Box::new(BigramLanguageModel::new(vocab_size, vb)?)),
            Model::Transformer => Ok(Box::new(TransformerLanguageModel::new(
                args.blocks,
                vocab_size,
                vb,
            )?)),
        }
    };

    let model = create_model(vb)?;

    if let Some(load) = &args.load {
        let load = normalize_safetensors_filename(load);
        println!("Loading weights from {load}.");
        varmap.load(load)?;
    }

    let params = ParamsAdamW {
        lr: args.lr,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    if args.vars {
        let data = varmap.data().lock().unwrap();
        println!("varmap vars: {:#?}", data);
    }

    let create_model_no_grad = || -> Result<Box<dyn Module>> {
        // "Freeze" the varmap as detached tensors to ensure that gradients aren't calculated
        // for our parameters. While this doesn't actually seem to improve performance, it _does_
        // seem to result in better training, since our evals don't mess with our optimizer: when
        // running with `--epochs=5000 --lr=1e-3 --blocks=1` the loss improves from 2.193 to 2.179
        // when using the no-gradient variant of the model for evals.
        let varmap_data = varmap.data().lock().unwrap();
        let mut detached_vars: HashMap<String, Tensor> = HashMap::with_capacity(varmap_data.len());
        for (path, var) in varmap_data.iter() {
            detached_vars.insert(path.clone(), var.as_detached_tensor());
        }
        create_model(VarBuilder::from_tensors(detached_vars, DType::F32, &device))
    };

    let estimate_loss = |data: &Tensor, rng: &mut StdRng| -> Result<f32> {
        let mut losses = Vec::with_capacity(EVAL_ITERS);
        let model_no_grad = create_model_no_grad()?;
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
            // the _variables_, not the inputs, which is why I wrote `create_model_no_grad`
            // above.
            let (xs, ys) = get_batch(&data, rng)?;
            let logits = model_no_grad.forward(&xs)?;
            let loss = language_loss(&logits, &ys)?;
            losses.push(loss.to_scalar()?);
        }
        Ok(losses.iter().sum::<f32>() / losses.len() as f32)
    };

    let calculate_loss = |prefix: String, rng: &mut StdRng| -> Result<()> {
        let train_loss = estimate_loss(&train_data, rng)?;
        let val_loss = estimate_loss(&val_data, rng)?;
        println!("{prefix} train loss {train_loss:.4}, val loss {val_loss:.4}",);
        Ok(())
    };

    let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    if args.epochs > 0 {
        calculate_loss("Initial".to_owned(), &mut rng)?;
    }
    for i in 1..=args.epochs {
        let (xs, ys) = get_batch(&train_data, &mut rng)?;
        let logits = model.forward(&xs)?;
        let loss = language_loss(&logits, &ys)?;
        let gradients = loss.backward()?;
        if args.vars && i == args.epochs {
            let data = varmap.data().lock().unwrap();
            for (name, var) in data.iter() {
                let tensor = var.deref();
                if let Some(grad) = gradients.get(tensor) {
                    let grad_squared = grad.sqr()?;
                    let grad_norm: f32 = grad_squared.sum_all()?.sqrt()?.to_scalar()?;
                    println!("gradient norm for {name}: {:.4}", grad_norm);
                    if grad_norm > 10.0 {
                        println!("  ⚠️  WARNING: Large gradient!");
                    } else if grad_norm < 1e-6 {
                        println!("  ⚠️  WARNING: Vanishing gradient!");
                    }
                } else {
                    println!("⚠️  WARNING: No gradient for {name}!");
                }
            }
        }
        optimizer.step(&gradients)?;

        if i % EVAL_INTERVAL == 0 || i == args.epochs {
            calculate_loss(format!("Epoch {i}"), &mut rng)?;
        }
    }

    if args.epochs > 0 {
        let end_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        println!("Total training time: {} ms", end_time - start_time)
    } else {
        calculate_loss("Model".to_owned(), &mut rng)?;
    }

    if let Some(save) = &args.save {
        let save = normalize_safetensors_filename(save);
        println!("Saving weights to {save}.");
        varmap.save(save)?;
    }

    if args.chars > 0 {
        let mut rng = StdRng::seed_from_u64(seed);
        let model_no_grad = create_model_no_grad()?;
        let result = language_generate(&model_no_grad, args.chars, &mut rng, &device)?;
        println!("{}", tokenizer.decode(&result)?);
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
