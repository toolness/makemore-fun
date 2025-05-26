mod tokenizer;

use candle_core::{DType, Device, IndexOp, Tensor};
use anyhow::Result;
use candle_nn::{Embedding, Module, VarBuilder, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizer::Tokenizer;

/// Number of examples in each batch.
const BATCH_SIZE: usize = 4;

/// Context size, in tokens.
const BLOCK_SIZE: usize = 8;

struct BigramLanguageModel {
    token_embedding_table: Embedding
}

impl BigramLanguageModel {
    fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, vocab_size, vb)?;
        Ok(Self { token_embedding_table })
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
    let mut rng = StdRng::seed_from_u64(123);
    let device = Device::Cpu;

    let tiny_shakespeare = get_tiny_shakespeare()?;
    let tokenizer = Tokenizer::from_string(&tiny_shakespeare)?;
    println!("Initialized tokenizer with {} tokens.", tokenizer.len());
    println!("encoded 'hii there': {:?}", tokenizer.encode("hii there")?);
    println!("decoded 'hii there': {:?}", tokenizer.decode(&tokenizer.encode("hii there")?)?);

    let data = Tensor::new(tokenizer.encode(&tiny_shakespeare)?, &device)?;
    let data_len = data.shape().dim(0)?;

    println!("Data is {} tokens.", data_len);
    let train_split_index = (0.9 * (data_len as f64)) as usize;
    let train_data = data.i(..train_split_index)?;
    let val_data = data.i(train_split_index..)?;
    println!("Training data is {} tokens.", train_data.shape().dim(0)?);
    println!("Validation data is {} tokens.", val_data.shape().dim(0)?);

    let get_batch = |data: Tensor, rng: &mut StdRng| -> Result<(Tensor, Tensor)> {
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

    let (xs, y) = get_batch(train_data, &mut rng)?;

    println!("xs:\n{xs}\ny:\n{y}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BigramLanguageModel::new(vb.clone(), tokenizer.len())?;
    let logits = model.forward(&xs)?;

    println!("logits shape: {:?}", logits.shape());
    assert_eq!(logits.dims3()?, (BATCH_SIZE, BLOCK_SIZE, tokenizer.len()));
    println!("logits:\n{logits}");

    Ok(())
}
