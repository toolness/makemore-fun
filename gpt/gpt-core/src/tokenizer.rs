use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};

use crate::{
    char_tokenizer::CharTokenizer, pair_tokenizers::CharPairTokenizer, util::SafetensorLoader,
};

/// This enum represents the different kinds of Tokenizers that can be
/// deserialized and provides methods to load them.
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

    /// Loads the tokenizer from its serialized form in the given safetensors
    /// object.
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
            TokenizerType::CharPair => {
                let tensor = safetensors.load_tensor(self.safetensors_key(), device)?;
                Ok(Box::new(CharPairTokenizer::from_tensor(&tensor)?))
            }
        }
    }

    /// Attempts to find any serialized Tokenizer in the given safetensors object.
    pub fn load_any<T: SafetensorLoader>(
        safetensors: &T,
        device: &Device,
    ) -> Result<Box<dyn Tokenizer>> {
        let all_types = &[TokenizerType::Char, TokenizerType::CharPair];
        for tokenizer_type in all_types {
            if let Ok(tokenizer) = tokenizer_type.load(safetensors, device) {
                return Ok(tokenizer);
            }
        }
        Err(anyhow!("Unable to load any tokenizer from safetensors!"))
    }
}

/// Represents a Tokenizer that converts strings to vectors of tokens
/// (represented as u32 integers) and back.
pub trait Tokenizer {
    /// The size of the Tokenizer's vocabulary.
    fn len(&self) -> usize;

    /// Attempts to tokenize the given string. If any characters in
    /// the string don't map to tokens in the vocabulary, an error
    /// will be returned.
    fn encode(&self, content: &str) -> Result<Vec<u32>>;

    /// Like `encode` but filters out any content that doesn't map to a
    /// token in the vocabulary.
    fn encode_lossy(&self, content: &str) -> Vec<u32>;

    /// Attempts to convert the given tokens into a string, returning
    /// an error if the tokens are out of range or don't map to a valid
    /// string.
    fn decode(&self, tokens: &Vec<u32>) -> Result<String>;

    /// Attempts to serialize the Tokenizer into a Tensor.
    ///
    /// This is an odd way of serializing a Tokenizer, but I was
    /// already saving the model into a safetensors file and wanted to
    /// be able to store the Tokenizer along with it, without needing
    /// to save a completely different file.
    ///
    /// Also, theoretically safetensors does support metadata strings, so we
    /// could serialize JSON into them, but Candle makes it difficult to
    /// access those, so I'm just using tensors for now.
    fn as_tensor(&self, device: &Device) -> Result<Tensor>;

    /// Returns the tokenizer type that corresponds to the Tokenizer.
    fn tokenizer_type(&self) -> TokenizerType;

    /// Attempts to return a string that describes the vocabulary of the
    /// Tokenizer as debugging ouput.
    fn debug_vocab(&self) -> String;
}
