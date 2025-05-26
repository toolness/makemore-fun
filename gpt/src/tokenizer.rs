use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};

pub struct Tokenizer {
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

    pub fn len(&self) -> usize {
        self.ctoi.len()
    }

    pub fn encode<T: AsRef<str>>(&self, content: T) -> Result<Vec<usize>> {
        let mut result = Vec::with_capacity(content.as_ref().len());

        for char in content.as_ref().chars() {
            let Some(token) = self.ctoi.get(&char) else {
                return Err(anyhow!("'{}' is not a valid character", char));
            };
            result.push(*token);
        }

        Ok(result)
    }

    pub fn decode(&self, tokens: &Vec<usize>) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string() {
        let input = String::from("abc");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

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
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        let encoded = tokenizer.encode("abc").unwrap();
        assert_eq!(encoded, vec![0, 1, 2]);

        let invalid_result = tokenizer.encode("d");
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_decode() {
        let input = String::from("acb");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        let decoded = tokenizer.decode(&vec![0, 1, 2]).unwrap();
        assert_eq!(decoded, "abc");

        let invalid_result = tokenizer.decode(&vec![3]);
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_len() {
        let input = String::from("abc");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3);
    }

    #[test]
    fn test_empty_string() {
        let input = String::from("");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 0);
        assert!(tokenizer.encode("").unwrap().is_empty());
        assert!(tokenizer.decode(&vec![]).unwrap().is_empty());
    }

    #[test]
    fn test_special_characters() {
        let input = String::from("!@#");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3);
        assert_eq!(tokenizer.encode("!@#").unwrap(), vec![0, 2, 1]);
        assert_eq!(tokenizer.decode(&vec![0, 2, 1]).unwrap(), "!@#");
    }

    #[test]
    fn test_repeated_characters() {
        let input = String::from("aabbcc");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        assert_eq!(tokenizer.len(), 3); // Only unique characters
        assert_eq!(tokenizer.encode("abc").unwrap(), vec![0, 1, 2]);
        assert_eq!(tokenizer.decode(&vec![0, 1, 2]).unwrap(), "abc");
    }

    #[test]
    fn test_invalid_decode() {
        let input = String::from("abc");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        let invalid_result = tokenizer.decode(&vec![99]);
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_partial_encode() {
        let input = String::from("abc");
        let tokenizer = Tokenizer::from_string(&input).unwrap();

        let partial_result = tokenizer.encode("abz");
        assert!(partial_result.is_err());
    }
}
