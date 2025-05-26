mod tokenizer;

use candle_core::{Device, Tensor};
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

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");

    let tiny_shakespeare = get_tiny_shakespeare()?;
    let tokenizer = Tokenizer::from_string(&tiny_shakespeare)?;
    println!("Initialized tokenizer with {} tokens.", tokenizer.len());
    println!("encoded 'hii there': {:?}", tokenizer.encode("hii there")?);
    println!("decoded 'hii there': {:?}", tokenizer.decode(&tokenizer.encode("hii there")?)?);

    Ok(())
}
