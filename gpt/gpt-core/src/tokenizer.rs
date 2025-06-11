use anyhow::Result;

pub trait Tokenizer {
    fn len(&self) -> usize;

    fn encode(&self, content: &str) -> Result<Vec<u32>>;

    /// Like `encode` but filters out any content that doesn't map to a
    /// token in the vocabulary.
    fn encode_lossy(&self, content: &str) -> Vec<u32>;

    fn decode(&self, tokens: &Vec<u32>) -> Result<String>;
}
