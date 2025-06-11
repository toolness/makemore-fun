use anyhow::Result;
use candle_core::{Device, Tensor};

/// Key in safetensors file to store tokenizer vocabulary.
/// Prefixing it with "BUFFER." because this is similar to a pytorch
/// buffer and we want to make it obvious that it's not a trainable
/// model parameter.
///
/// The key's value is meant to be a Tensor returned by `Tokenizer::into_tensor()`.
pub const TOKENIZER_VOCABULARY_KEY: &'static str = "BUFFER.tokenizer_vocabulary";

pub trait Tokenizer {
    fn len(&self) -> usize;

    fn encode(&self, content: &str) -> Result<Vec<u32>>;

    /// Like `encode` but filters out any content that doesn't map to a
    /// token in the vocabulary.
    fn encode_lossy(&self, content: &str) -> Vec<u32>;

    fn decode(&self, tokens: &Vec<u32>) -> Result<String>;

    fn as_tensor(&self, device: &Device) -> Result<Tensor>;

    fn debug_vocab(&self) -> String;
}
