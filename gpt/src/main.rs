mod tokenizer;

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Module, Optimizer, VarBuilder, VarMap, loss::cross_entropy, ops::softmax,
};
use candle_optimisers::adam::{Adam, ParamsAdam};
use clap::Parser;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};
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

struct BigramLanguageModel {
    vocab_size: usize,
    token_embedding_table: Embedding,
}

impl BigramLanguageModel {
    fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, vocab_size, vb)?;
        Ok(Self {
            vocab_size,
            token_embedding_table,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.token_embedding_table.forward(xs)?)
    }

    fn loss(&self, logits: &Tensor, ys: &Tensor) -> Result<Tensor> {
        assert_eq!(logits.dims3()?, (BATCH_SIZE, BLOCK_SIZE, self.vocab_size));

        if cfg!(debug_assertions) {
            let sm = softmax(&logits, D::Minus1)?;
            assert_relative_eq!(
                sm.get(0)?.get(0)?.sum(0)?.to_scalar::<f32>()?,
                1.0,
                epsilon = 0.0001
            );
        }

        let flat_logits = logits.reshape((BATCH_SIZE * BLOCK_SIZE, self.vocab_size))?;
        let flat_ys = ys.reshape(BATCH_SIZE * BLOCK_SIZE)?;

        if cfg!(debug_assertions) {
            assert_equal_tensors(logits.get(0)?.get(0)?, flat_logits.get(0)?)?;
        }
        let loss = cross_entropy(&flat_logits, &flat_ys)?;
        Ok(loss)
    }

    fn generate(&self, num_chars: usize, rng: &mut StdRng, device: &Device) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(num_chars);
        result.push(0);
        for _ in 0..num_chars {
            let block_slice = &result[result.len().saturating_sub(BLOCK_SIZE)..];
            let block = Tensor::from_slice(block_slice, (1, block_slice.len()), device)?;
            let logits = self.forward(&block)?;
            // Take just the logits for the final time step.
            let logits = logits.i((.., block_slice.len() - 1, ..))?;
            let sm = softmax(&logits, 1)?;
            let token = multinomial(&sm, rng)?;
            result.push(token);
        }
        Ok(result)
    }
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
    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(timestamp));
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

    let result = model.generate(100, &mut rng, &device)?;
    println!("{:?}", tokenizer.decode(&result)?);

    Ok(())
}

/// Uh, candle doesn't seem to have multinomial sampling built-in, so
/// we'll just implement something janky here.
///
/// We could consider using https://github.com/EricLBuehler/candle-sampling
/// instead.
fn multinomial(tensor: &Tensor, rng: &mut StdRng) -> Result<u32> {
    let vec: Vec<f32> = tensor.get(0)?.to_vec1()?;
    let mut choices: Vec<u32> = Vec::with_capacity(vec.len());
    let mut weights: Vec<u32> = Vec::with_capacity(vec.len());

    for (i, prob) in vec.iter().enumerate() {
        let weight = (prob * 100.0) as u32;
        if weight > 0 {
            choices.push(i as u32);
            weights.push(weight);
        }
    }

    let dist = WeightedIndex::new(&weights)?;

    Ok(choices[dist.sample(rng)])
}

/// Uh, candle doesn't have an easy way of comparing tensors for
/// equality so we'll do this.
fn assert_equal_tensors(a: Tensor, b: Tensor) -> Result<()> {
    // WHY IS THIS SO HARD????????????????
    let eq = a.eq(&b)?.flatten_all()?.to_vec1::<u8>()?;
    for item in eq {
        assert_eq!(item, 1);
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
