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
