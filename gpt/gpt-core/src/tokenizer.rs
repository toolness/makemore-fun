use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::{char_tokenizer::CharTokenizer, util::SafetensorLoader};

pub enum TokenizerType {
    Char,
    CharPair,
}

impl TokenizerType {
    /// Key in safetensors file to store tokenizer vocabulary.
    /// Prefixing it with "BUFFER." because this is similar to a pytorch
    /// buffer and we want to make it obvious that it's not a trainable
    /// model parameter.
    ///
    /// The key's value is meant to be a Tensor returned by `Tokenizer::as_tensor()`.
    pub fn safetensors_key(&self) -> &'static str {
        match self {
            // This doesn't have the word 'char' in it b/c it was made before we
            // had multiple tokenizers, and we want to remain backwards-compatible.
            TokenizerType::Char => "BUFFER.tokenizer_vocabulary",
            TokenizerType::CharPair => "BUFFER.char_pair_tokenizer_vocabulary",
        }
    }

    pub fn load<T: SafetensorLoader>(
        &self,
        safetensors: &T,
        device: &Device,
    ) -> Result<Box<dyn Tokenizer>> {
        match self {
            TokenizerType::Char => {
                let tensor = safetensors.load_tensor(self.safetensors_key(), device)?;
                Ok(Box::new(CharTokenizer::from_tensor(&tensor)?))
            }
            TokenizerType::CharPair => todo!(),
        }
    }
}

pub trait Tokenizer {
    fn len(&self) -> usize;

    fn encode(&self, content: &str) -> Result<Vec<u32>>;

    /// Like `encode` but filters out any content that doesn't map to a
    /// token in the vocabulary.
    fn encode_lossy(&self, content: &str) -> Vec<u32>;

    fn decode(&self, tokens: &Vec<u32>) -> Result<String>;

    fn as_tensor(&self, device: &Device) -> Result<Tensor>;

    fn tokenizer_type(&self) -> TokenizerType;

    fn debug_vocab(&self) -> String;
}
