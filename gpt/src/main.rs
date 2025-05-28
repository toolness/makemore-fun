mod tokenizer;

use clap::Parser;
use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Module, Optimizer, VarBuilder, VarMap, loss::cross_entropy, ops::softmax,
};
use candle_optimisers::adam::{Adam, ParamsAdam};
use rand::{distr::{weighted::WeightedIndex, Distribution}, rngs::StdRng, Rng, SeedableRng};
use tokenizer::Tokenizer;

/// Number of examples in each batch.
const BATCH_SIZE: usize = 32;

/// Context size, in tokens.
const BLOCK_SIZE: usize = 8;

const LEARNING_RATE: f64 = 0.001;

#[derive(Parser)]
pub struct Args {
    #[arg(long, default_value_t = 10_000)]
    pub epochs: usize,
}

struct BigramLanguageModel {
    token_embedding_table: Embedding,
}

impl BigramLanguageModel {
    fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, vocab_size, vb)?;
        Ok(Self {
            token_embedding_table,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.token_embedding_table.forward(xs)?)
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
    let mut rng = StdRng::seed_from_u64(123);
    let device = Device::Cpu;

    let tiny_shakespeare = get_tiny_shakespeare()?;
    let tokenizer = Tokenizer::from_string(&tiny_shakespeare)?;
    let vocab_size = tokenizer.len();
    println!("Initialized tokenizer with {} tokens.", vocab_size);
    println!("encoded 'hii there': {:?}", tokenizer.encode("hii there")?);
    println!(
        "decoded 'hii there': {:?}",
        tokenizer.decode(&tokenizer.encode("hii there")?)?
    );

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

    let (xs, ys) = get_batch(&train_data, &mut rng)?;

    println!("xs:\n{xs}\nys:\n{ys}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BigramLanguageModel::new(vb.clone(), vocab_size)?;
    let params = ParamsAdam {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut sgd = Adam::new(varmap.all_vars(), params)?;
    println!("varmap vars: {:?}", varmap.all_vars());

    for i in 0..=args.epochs {
        let (xs, ys) = get_batch(&train_data, &mut rng)?;
        let logits = model.forward(&xs)?;

        assert_eq!(logits.dims3()?, (BATCH_SIZE, BLOCK_SIZE, vocab_size));

        if cfg!(debug_assertions) {
            let sm = softmax(&logits, D::Minus1)?;
            assert_relative_eq!(
                sm.get(0)?.get(0)?.sum(0)?.to_scalar::<f32>()?,
                1.0,
                epsilon = 0.0001
            );
        }

        let flat_logits = logits.reshape((BATCH_SIZE * BLOCK_SIZE, vocab_size))?;
        let flat_ys = ys.reshape(BATCH_SIZE * BLOCK_SIZE)?;

        if cfg!(debug_assertions) {
            assert_equal_tensors(logits.get(0)?.get(0)?, flat_logits.get(0)?)?;
        }
        let loss = cross_entropy(&flat_logits, &flat_ys)?;
        sgd.backward_step(&loss)?;

        if i % 100 == 0 {
            println!("loss at epoch {i}: {}", loss.to_scalar::<f32>()?);
        }
    }

    let num_chars: usize = 100;
    let mut token: u32 = 0;
    let mut result = Vec::with_capacity(num_chars);
    for _ in 0..num_chars {
        let data: [u32; 1] = [token];
        let block = Tensor::from_slice(&data, (1,), &device)?;
        let logits = model.forward(&block)?;
        let sm = softmax(&logits, 1)?;
        token = multinomial(&sm, &mut rng)?;
        result.push(token);
    }

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
