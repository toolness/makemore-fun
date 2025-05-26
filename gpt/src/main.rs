mod tokenizer;

use candle_core::{Device, IndexOp, Tensor};
use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokenizer::Tokenizer;

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

    let batch_size = 4;
    let block_size = 8;

    let get_batch = |data: Tensor, rng: &mut StdRng| -> Result<(Tensor, Tensor)> {
        let mut x = Vec::with_capacity(batch_size);
        let mut y = Vec::with_capacity(batch_size);
        let data_len = data.shape().dim(0)?;
        for _ in 0..batch_size {
            // Ideally we'd use candle for these random numbers, but as far as I can tell,
            // it can only generate random floats. I guess we could round/cast them to
            // integers but for now I'm just going to use the rand crate instead.
            let idx: usize = rng.random_range(0..(data_len - block_size));
            x.push(data.i(idx..(block_size + idx))?);
            y.push(data.i((idx + 1)..(block_size + idx + 1))?);
        }
        Ok((Tensor::stack(&x, 0)?, Tensor::stack(&y, 0)?))
    };

    let (x, y) = get_batch(train_data, &mut rng)?;

    println!("x:\n{x}\ny:\n{y}");

    Ok(())
}
