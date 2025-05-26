mod tokenizer;

use candle_core::{Device, IndexOp, Tensor};
use anyhow::Result;
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

    Ok(())
}
