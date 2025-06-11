use std::{
    char::CharTryFromError,
    collections::{HashMap, HashSet},
};

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};

use crate::tokenizer::Tokenizer;

/// Character-level tokenizer in the style of Karpathy's
/// neural net lectures.
#[derive(Clone)]
pub struct CharTokenizer {
    ctoi: HashMap<char, u32>,
    itoc: HashMap<u32, char>,
}

impl CharTokenizer {
    pub fn from_char_vec(mut vec: Vec<char>) -> Result<Self> {
        vec.sort();
        let mut ctoi: HashMap<char, u32> = HashMap::new();
        let mut itoc = HashMap::new();
        for (i, char) in vec.iter().enumerate() {
            ctoi.insert(*char, i as u32);
            itoc.insert(i as u32, *char);
        }
        let result = CharTokenizer { ctoi, itoc };

        Ok(result)
    }

    pub fn into_char_vec(self) -> Vec<char> {
        let mut chars: Vec<char> = self.ctoi.into_keys().collect();
        chars.sort();
        chars
    }

    pub fn from_string(string: &String) -> Result<Self> {
        let mut all_chars: HashSet<char> = HashSet::new();
        all_chars.extend(string.chars());
        CharTokenizer::from_char_vec(all_chars.iter().copied().collect())
    }

    /// Given a one-dimensional tensor with each character representing a
    /// unicode scalar, returns the tokenizer for it.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let chars: Result<Vec<char>, CharTryFromError> = tensor
            .to_vec1::<u32>()?
            .iter()
            .map(|&u32| char::try_from(u32))
            .collect();
        Ok(CharTokenizer::from_char_vec(chars?)?)
    }

    pub fn decode_char(&self, token: u32) -> Result<char> {
        let Some(&char) = self.itoc.get(&token) else {
            return Err(anyhow!("'{}' is not a valid token", token));
        };
        Ok(char)
    }
}

impl Tokenizer for CharTokenizer {
    fn len(&self) -> usize {
        self.ctoi.len()
    }

    fn encode(&self, content: &str) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(content.len());

        for char in content.chars() {
            let Some(token) = self.ctoi.get(&char) else {
                return Err(anyhow!("'{}' is not a valid character", char));
            };
            result.push(*token);
        }

        Ok(result)
    }

    fn encode_lossy(&self, content: &str) -> Vec<u32> {
        let mut result = Vec::with_capacity(content.len());

        for char in content.chars() {
            if let Some(&token) = self.ctoi.get(&char) {
                result.push(token);
            };
        }

        result
    }

    fn decode(&self, tokens: &Vec<u32>) -> Result<String> {
        let mut result = String::with_capacity(tokens.len());

        for &token in tokens.iter() {
            result.push(self.decode_char(token)?);
        }

        Ok(result)
    }

    /// Returns a one-dimensional tensor with each character in the vocabulary
    /// represented by a unicode scalar.
    fn as_tensor(&self, device: &Device) -> Result<Tensor> {
        let len = self.ctoi.len();
        let vec = self
            .clone()
            .into_char_vec()
            .into_iter()
            .map(|char| char as u32)
            .collect();
        Ok(Tensor::from_vec(vec, (len,), device)?)
    }

    fn tokenizer_type(&self) -> crate::tokenizer::TokenizerType {
        crate::tokenizer::TokenizerType::Char
    }

    fn debug_vocab(&self) -> String {
        let str: String = self.clone().into_char_vec().iter().collect();
        format!("{:?}", str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3);
        assert_eq!(tokenizer.ctoi.get(&'a'), Some(&0));
        assert_eq!(tokenizer.ctoi.get(&'b'), Some(&1));
        assert_eq!(tokenizer.ctoi.get(&'c'), Some(&2));
        assert_eq!(tokenizer.itoc.get(&0), Some(&'a'));
        assert_eq!(tokenizer.itoc.get(&1), Some(&'b'));
        assert_eq!(tokenizer.itoc.get(&2), Some(&'c'));
    }

    #[test]
    fn test_encode() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        let encoded = tokenizer.encode("abc").unwrap();
        assert_eq!(encoded, vec![0, 1, 2]);

        let invalid_result = tokenizer.encode("d");
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_decode() {
        let input = String::from("acb");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        let decoded = tokenizer.decode(&vec![0, 1, 2]).unwrap();
        assert_eq!(decoded, "abc");

        let invalid_result = tokenizer.decode(&vec![3]);
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_len() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3);
    }

    #[test]
    fn test_empty_string() {
        let input = String::from("");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 0);
        assert!(tokenizer.encode("").unwrap().is_empty());
        assert!(tokenizer.decode(&vec![]).unwrap().is_empty());
    }

    #[test]
    fn test_special_characters() {
        let input = String::from("!@#");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3);
        assert_eq!(tokenizer.encode("!@#").unwrap(), vec![0, 2, 1]);
        assert_eq!(tokenizer.decode(&vec![0, 2, 1]).unwrap(), "!@#");
    }

    #[test]
    fn test_repeated_characters() {
        let input = String::from("aabbcc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3); // Only unique characters
        assert_eq!(tokenizer.encode("abc").unwrap(), vec![0, 1, 2]);
        assert_eq!(tokenizer.decode(&vec![0, 1, 2]).unwrap(), "abc");
    }

    #[test]
    fn test_invalid_decode() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        let invalid_result = tokenizer.decode(&vec![99]);
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_partial_encode() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();

        let partial_result = tokenizer.encode("abz");
        assert!(partial_result.is_err());
    }

    #[test]
    fn test_into_char_vec_basic() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();
        let chars = tokenizer.into_char_vec();
        assert_eq!(chars, vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_into_char_vec_empty() {
        let input = String::from("");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();
        let chars = tokenizer.into_char_vec();
        assert!(chars.is_empty());
    }

    #[test]
    fn test_into_char_vec_special_characters() {
        let input = String::from("!@#");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();
        let chars = tokenizer.into_char_vec();
        assert_eq!(chars, vec!['!', '#', '@']);
    }

    #[test]
    fn test_into_char_vec_repeated_characters() {
        let input = String::from("aabbcc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();
        let chars = tokenizer.into_char_vec();
        assert_eq!(chars, vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_from_into_tensor_works() {
        let input = String::from("abc");
        let tokenizer = CharTokenizer::from_string(&input).unwrap();
        let tensor = tokenizer.as_tensor(&Device::Cpu).unwrap();
        assert_eq!(tensor.to_vec1::<u32>().unwrap(), vec![97, 98, 99]);
        let tokenizer = CharTokenizer::from_tensor(&tensor).unwrap();
        assert_eq!(tokenizer.into_char_vec(), vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_trait_object_works() {
        let _trait_obj: Box<dyn Tokenizer> =
            Box::new(CharTokenizer::from_string(&"hi".to_owned()).unwrap());
    }
}
