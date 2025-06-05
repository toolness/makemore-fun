mod args;
mod device;

use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use args::Args;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use gpt_core::tokenizer::Tokenizer;
use gpt_core::util::{count_params, print_gradient_info};
use gpt_core::{
    language_model::{LanguageGenerator, language_loss},
    tokenizer::TOKENIZER_VOCABULARY_KEY,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng, rngs::StdRng};

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

    let mut safetensors_tokenizer: Option<Tokenizer> = None;

    let safetensors = if let Some(load) = &args.load {
        let load = normalize_safetensors_filename(load);
        println!("Loading weights from {load}.");

        // This is taken from VarMap::load(), it's too bad it's unsafe.
        // Actually it is kind of funny that we're loading SAFE tensors in
        // an UNSAFE block...
        let data = unsafe { candle_core::safetensors::MmapedSafetensors::new(load)? };

        if let Ok(tokenizer_tensor) = data.load(TOKENIZER_VOCABULARY_KEY, &device) {
            safetensors_tokenizer = Some(Tokenizer::from_tensor(&tokenizer_tensor)?);
        }
        Some(data)
    } else {
        None
    };

    let mut training_info: Option<(usize, Tensor)> = None;
    let mut training_tokenizer: Option<Tokenizer> = None;

    if args.epochs > 0 || safetensors_tokenizer.is_none() {
        let (tokenizer, training_data) =
            generate_training_data(std::fs::read_to_string(&args.corpus)?, &device)?;
        if args.epochs > 0 {
            training_info = Some((args.epochs, training_data));
        }
        training_tokenizer = Some(tokenizer);
    }

    let tokenizer = match (safetensors_tokenizer, training_tokenizer) {
        (None, None) => {
            return Err(anyhow!("Unable to load tokenizer!"));
        }
        (None, Some(tokenizer)) => tokenizer,
        (Some(tokenizer), None) => tokenizer,
        (Some(safetensors_tokenizer), Some(training_tokenizer)) => {
            if safetensors_tokenizer.len() != training_tokenizer.len() {
                // Return an error, the model has been trained assuming one vocabulary
                // size, we can't train it with a different vocabulary size.
                return Err(anyhow!(
                    "Mismatch between tokenizer data in safetensors file and training data!"
                ));
            }
            safetensors_tokenizer
        }
    };

    let context = tokenizer.encode(&args.context)?;
    let vocab_size = tokenizer.len();
    println!("Initialized tokenizer with {} tokens.", vocab_size);

    // let (xs, ys) = get_batch(&train_data, &mut rng)?;
    // println!("xs:\n{xs}\nys:\n{ys}");

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = args.create_model(vocab_size, vb)?;

    println!("Parameters in model: {}", count_params(&varmap));

    if let Some(data) = safetensors {
        // This is mostly what VarMap::load() does, only we've already got thee
        // safetensors file loaded already, so we're just going to populate the
        // model params from it.
        let mut tensor_data = varmap.data().lock().unwrap();
        for (name, var) in tensor_data.iter_mut() {
            let data = data.load(name, var.device())?;
            if let Err(err) = var.set(&data) {
                return Err(anyhow!(
                    "error setting {name} using data from {:?}: {err}",
                    args.load
                ));
            }
        }
    }

    if args.vars {
        let data = varmap.data().lock().unwrap();
        println!("varmap vars: {:#?}", data);
    }

    if let Some((epochs, training_data)) = training_info {
        let start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let main_pb = multi_progress.add(ProgressBar::new(epochs as u64));
        main_pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        let trainer = Trainer::new(training_data, args.batch_size, args.block_size)?;

        trainer.estimate_loss(
            "Initial".to_owned(),
            &mut rng,
            &multi_progress,
            &args.create_model_no_grad(vocab_size, &varmap, &device)?,
        )?;
        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: args.lr,
                ..Default::default()
            },
        )?;
        for i in 1..=epochs {
            main_pb.set_message("Training");
            let (xs, ys) = trainer.get_batch(TrainingSet::Train, &mut rng)?;
            let logits = model.forward(&xs)?;
            let loss = language_loss(&logits, &ys)?;
            let gradients = loss.backward()?;
            if args.vars && i == epochs {
                print_gradient_info(&varmap, &gradients)?;
            }
            optimizer.step(&gradients)?;
            main_pb.inc(1);

            if i % EVAL_INTERVAL == 0 || i == epochs {
                trainer.estimate_loss(
                    format!("Epoch {i}"),
                    &mut rng,
                    &multi_progress,
                    &args.create_model_no_grad(vocab_size, &varmap, &device)?,
                )?;
            }
        }

        let end_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        println!("Total training time: {} ms", end_time - start_time)
    }

    if let Some(save) = &args.save {
        let save = normalize_safetensors_filename(save);
        println!("Saving weights to {save}.");
        // We need to get the key, which creates it, before we can actually
        // set it (this feels very weird).
        //
        // Note that this mutates the varmap, which I don't really like,
        // because e.g. our gradient inspection tools might think it's a
        // real variable, when it's actually not. But this is right before
        // we perform inference anyways so it's not that big a deal.
        varmap.get(
            (tokenizer.len(),),
            TOKENIZER_VOCABULARY_KEY,
            candle_nn::Init::Const(0.0),
            DType::U32,
            &device,
        )?;
        varmap.set_one(
            TOKENIZER_VOCABULARY_KEY,
            tokenizer.clone().into_tensor(&device)?,
        )?;
        varmap.save(save)?;
    }

    if args.chars > 0 {
        let mut rng = StdRng::seed_from_u64(seed);
        let model_no_grad = args.create_model_no_grad(vocab_size, &varmap, &device)?;
        print!("{}", args.context);
        let mut language_generator =
            LanguageGenerator::new(&context, model_no_grad, args.block_size)?;
        for _ in 0..args.chars {
            let char =
                language_generator.next_char(&mut rng, &tokenizer, args.temperature, &device)?;
            print!("{}", char);
            std::io::stdout().flush()?;
        }
        println!("");
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

#[derive(Clone, Copy)]
pub enum TrainingSet {
    Train,
    Val,
}

fn generate_training_data(
    training_corpus: String,
    device: &candle_core::Device,
) -> Result<(Tokenizer, Tensor)> {
    let tokenizer = Tokenizer::from_string(&training_corpus)?;
    let data = Tensor::new(tokenizer.encode(&training_corpus)?, &device)?;
    Ok((tokenizer, data))
}

pub struct Trainer {
    batch_size: usize,
    block_size: usize,
    train_data: Tensor,
    val_data: Tensor,
}

impl Trainer {
    pub fn new(data: Tensor, batch_size: usize, block_size: usize) -> Result<Self> {
        let data_len = data.shape().dim(0)?;
        println!("Data is {} tokens.", data_len);
        let train_split_index = (0.9 * (data_len as f64)) as usize;
        let train_data = data.i(..train_split_index)?;
        let val_data = data.i(train_split_index..)?;
        println!("Training data is {} tokens.", train_data.shape().dim(0)?);
        println!("Validation data is {} tokens.", val_data.shape().dim(0)?);
        Ok(Self {
            batch_size,
            block_size,
            train_data,
            val_data,
        })
    }

    fn get_dataset(&self, training_set: TrainingSet) -> &Tensor {
        match training_set {
            TrainingSet::Train => &self.train_data,
            TrainingSet::Val => &self.val_data,
        }
    }

    pub fn get_batch(
        &self,
        training_set: TrainingSet,
        rng: &mut StdRng,
    ) -> Result<(Tensor, Tensor)> {
        let data = self.get_dataset(training_set);
        let mut x = Vec::with_capacity(self.batch_size);
        let mut y = Vec::with_capacity(self.batch_size);
        let data_len = data.shape().dim(0)?;
        for _ in 0..self.batch_size {
            // Ideally we'd use candle for these random numbers, but as far as I can tell,
            // it can only generate random floats. I guess we could round/cast them to
            // integers but for now I'm just going to use the rand crate instead.
            //
            // Also, I think this might be doing a lot of back-and-forth between GPU and CPU.
            // If we can use Candle for all of this, and only use Tensors, rather than pulling
            // from the data Tensor into a Vec and then converting it back into a Tensor,
            // this might be able to run faster on GPUs.
            let idx: usize = rng.random_range(0..(data_len - self.block_size));
            x.push(data.i(idx..(self.block_size + idx))?);
            y.push(data.i((idx + 1)..(self.block_size + idx + 1))?);
        }
        Ok((Tensor::stack(&x, 0)?, Tensor::stack(&y, 0)?))
    }

    pub fn estimate_loss(
        &self,
        prefix: String,
        rng: &mut StdRng,
        multi_progress: &MultiProgress,
        model_no_grad: &Box<dyn Module>,
    ) -> Result<()> {
        let estimate_dataset_loss = |training_set: TrainingSet, rng: &mut StdRng| -> Result<f32> {
            let mut losses = Vec::with_capacity(EVAL_ITERS);
            let bar = multi_progress.add(ProgressBar::new(EVAL_ITERS as u64));
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:30.white/white} {pos:>7}/{len:7} {msg}",
                )
                .unwrap()
                .progress_chars("##-"),
            );
            let name = match training_set {
                TrainingSet::Train => "train loss",
                TrainingSet::Val => "val loss",
            };
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
                let (xs, ys) = self.get_batch(training_set, rng)?;
                let logits = model_no_grad.forward(&xs)?;
                let loss = language_loss(&logits, &ys)?;
                losses.push(loss.to_scalar()?);
                bar.inc(1);
            }
            let loss = losses.iter().sum::<f32>() / losses.len() as f32;
            bar.finish_and_clear();
            Ok(loss)
        };

        let train_loss = estimate_dataset_loss(TrainingSet::Train, rng)?;
        let val_loss = estimate_dataset_loss(TrainingSet::Val, rng)?;
        println!("{prefix} train loss {train_loss:.4}, val loss {val_loss:.4}",);
        Ok(())
    }
}
