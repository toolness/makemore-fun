mod bigram_language_model;
mod language_model;
mod tokenizer;
mod util;

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use bigram_language_model::BigramLanguageModel;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use clap::Parser;
use language_model::LanguageModel;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokenizer::Tokenizer;

/// Number of examples in each batch.
const BATCH_SIZE: usize = 32;

/// Context size, in tokens.
const BLOCK_SIZE: usize = 8;

const LEARNING_RATE: f64 = 1e-2;

/// After how many epochs do we evaluate the model again?
const EVAL_INTERVAL: usize = 300;

/// How many batches to compute loss over.
const EVAL_ITERS: usize = 200;

/// Number of characters to generate.
const GENERATE_NUM_CHARS: usize = 500;

#[derive(Parser)]
pub struct Args {
    /// Random number seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of epochs to train the model.
    #[arg(long, default_value_t = 3_000)]
    pub epochs: usize,

    /// The file to save the trained weights to, in safetensors format.
    #[arg(long)]
    pub save: Option<String>,

    /// The file to load the trained weights from, in safetensors format.
    #[arg(long)]
    pub load: Option<String>,
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
    let now = SystemTime::now();
    let timestamp = now.duration_since(UNIX_EPOCH)?.as_secs();
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
    let model = BigramLanguageModel::new(vb.clone(), vocab_size)?;

    if let Some(load) = &args.load {
        let load = normalize_safetensors_filename(load);
        println!("Loading weights from {load}.");
        varmap.load(load)?;
    }

    let params = ParamsAdam {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut sgd = Adam::new(varmap.all_vars(), params)?;
    println!("varmap vars: {:?}", varmap.all_vars());

    let estimate_loss = |data: &Tensor, rng: &mut StdRng| -> Result<f32> {
        let mut losses = Vec::with_capacity(EVAL_ITERS);
        for _ in 0..EVAL_ITERS {
            let (xs, ys) = get_batch(&data, rng)?;
            let logits = model.forward(&xs)?;
            let loss = model.loss(&logits, &ys)?;
            losses.push(loss.to_scalar()?);
        }
        Ok(losses.iter().sum::<f32>() / losses.len() as f32)
    };

    for i in 0..=args.epochs {
        let (xs, ys) = get_batch(&train_data, &mut rng)?;
        let logits = model.forward(&xs)?;
        let loss = model.loss(&logits, &ys)?;
        sgd.backward_step(&loss)?;

        if i % EVAL_INTERVAL == 0 {
            let train_loss = estimate_loss(&train_data, &mut rng)?;
            let val_loss = estimate_loss(&val_data, &mut rng)?;
            println!("epoch {i}: train loss {train_loss:.4}, val loss {val_loss:.4}",);
        }
    }

    if let Some(save) = &args.save {
        let save = normalize_safetensors_filename(save);
        println!("Saving weights to {save}.");
        varmap.save(save)?;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let result = model.generate(GENERATE_NUM_CHARS, &mut rng, &device)?;
    println!("{}", tokenizer.decode(&result)?);

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
