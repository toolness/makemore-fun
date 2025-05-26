use std::collections::{HashMap, HashSet};

use candle_core::{Device, Tensor};
use anyhow::{anyhow, Result};

fn get_tiny_shakespeare() -> Result<String> {
    // https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    let content = std::fs::read_to_string("tiny-shakespeare.txt")?;
    Ok(content)
}

struct Tokenizer {
    ctoi: HashMap<char, usize>,
    itoc: HashMap<usize, char>,
}

impl Tokenizer {
    pub fn from_string(string: &String) -> Result<Self> {
        let mut all_chars: HashSet<char> = HashSet::new();
        all_chars.extend(string.chars());
        let mut all_chars_sorted: Vec<char> = all_chars.iter().copied().collect();
        all_chars_sorted.sort();
        let mut ctoi: HashMap<char, usize> = HashMap::new();
        let mut itoc = HashMap::new();
        for (i, char, ) in all_chars_sorted.iter().enumerate() {
            ctoi.insert(*char, i);
            itoc.insert(i, *char);
        }
        let result = Tokenizer {
            ctoi,
            itoc,
        };

        Ok(result)
    }

    fn len(&self) -> usize {
        self.ctoi.len()
    }

    fn encode<T: AsRef<str>>(&self, content: T) -> Result<Vec<usize>> {
        let mut result = Vec::with_capacity(content.as_ref().len());

        for char in content.as_ref().chars() {
            let Some(token) = self.ctoi.get(&char) else {
                return Err(anyhow!("'{}' is not a valid character", char));
            };
            result.push(*token);
        }

        Ok(result)
    }

    fn decode(&self, tokens: &Vec<usize>) -> Result<String> {
        let mut result = String::with_capacity(tokens.len());

        for token in tokens.iter() {
            let Some(char) = self.itoc.get(&token) else {
                return Err(anyhow!("'{}' is not a valid token", token));
            };
            result.push(*char);
        }

        Ok(result)
    }
}

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
