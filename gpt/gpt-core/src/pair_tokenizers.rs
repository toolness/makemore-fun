use std::{cmp::Ordering, collections::HashMap, iter::zip};

use anyhow::{Result, anyhow};

use crate::{char_tokenizer::CharTokenizer, tokenizer::Tokenizer};

pub fn merge(tokens: &[u32], pair: (u32, u32), pair_token_id: u32) -> Vec<u32> {
    let mut new_tokens = Vec::with_capacity(tokens.len());
    let mut i = 0;

    loop {
        if i >= tokens.len() {
            break;
        }
        if i < tokens.len() - 1 {
            let curr_pair = (tokens[i], tokens[i + 1]);
            if curr_pair == pair {
                new_tokens.push(pair_token_id);
                i += 2;
                continue;
            }
        }
        new_tokens.push(tokens[i]);
        i += 1;
    }

    new_tokens
}

fn get_most_common_pair(tokens: &[u32]) -> Result<(u32, u32)> {
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    if tokens.len() < 2 {
        return Err(anyhow!("tokens does not contain any pairs!"));
    }
    for (&a, &b) in zip(tokens.iter(), tokens[1..].iter()) {
        let pair = (a, b);
        let entry = counts.entry(pair).or_insert(0);
        *entry += 1;
    }

    let mut all = counts.into_iter().collect::<Vec<_>>();
    all.sort_by(|((a1, a2), a_count), ((b1, b2), b_count)| {
        let count_ordering = b_count.cmp(a_count);
        // This isn't semantically meaningful, but helps ensure a stable sort.
        if count_ordering == Ordering::Equal {
            let first_ordering = a1.cmp(b1);
            if first_ordering == Ordering::Equal {
                a2.cmp(b2)
            } else {
                first_ordering
            }
        } else {
            count_ordering
        }
    });

    Ok(all[0].0)
}

fn pair_compress(mut tokens: Vec<u32>, pair_to_token_map: &HashMap<(u32, u32), u32>) -> Vec<u32> {
    // This is pretty inefficient and can probably be improved a lot.
    loop {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;
        let mut keep_going = false;
        loop {
            if i >= tokens.len() {
                break;
            }
            if i < tokens.len() - 1 {
                let curr_pair = (tokens[i], tokens[i + 1]);
                if let Some(&token) = pair_to_token_map.get(&curr_pair) {
                    new_tokens.push(token);
                    i += 2;
                    keep_going = true;
                    continue;
                }
            }
            new_tokens.push(tokens[i]);
            i += 1;
        }
        tokens = new_tokens;
        if !keep_going {
            break;
        }
    }
    tokens
}

/// This is based on the byte-pair tokenizer from Karpathy's
/// "Let's build the GPT Tokenizer" lecture at:
///
///   https://www.youtube.com/watch?v=zduSFxRajkE
pub struct BytePairTokenizer {
    pair_to_token_map: HashMap<(u32, u32), u32>,
    token_to_bytes_map: HashMap<u32, Vec<u8>>,
}

impl BytePairTokenizer {
    pub fn new<T: AsRef<str>>(corpus: T, vocab_size: usize) -> Result<Self> {
        if vocab_size < 256 {
            return Err(anyhow!("vocab_size must be at least 256!"));
        }

        let mut tokens = corpus
            .as_ref()
            .as_bytes()
            .iter()
            .map(|&u8| u8 as u32)
            .collect::<Vec<_>>();

        let mut curr_vocab_size = 256;
        let mut pair_to_token_map = HashMap::new();
        let mut token_to_bytes_map = HashMap::new();

        for i in 0..=255u8 {
            token_to_bytes_map.insert(i as u32, vec![i]);
        }

        while curr_vocab_size < vocab_size {
            let pair = get_most_common_pair(&tokens)?;
            let new_token_id = curr_vocab_size as u32;
            curr_vocab_size += 1;
            pair_to_token_map.insert(pair, new_token_id);
            let new_token_bytes = token_to_bytes_map
                .get(&pair.0)
                .unwrap()
                .iter()
                .chain(token_to_bytes_map.get(&pair.1).unwrap().iter())
                .copied()
                .collect::<Vec<_>>();
            token_to_bytes_map.insert(new_token_id, new_token_bytes);
            //.extend(token_to_bytes_map.get(&pair.1).unwrap());
            tokens = merge(&tokens, pair, new_token_id);
        }

        Ok(Self {
            pair_to_token_map,
            token_to_bytes_map,
        })
    }
}

impl Tokenizer for BytePairTokenizer {
    fn len(&self) -> usize {
        self.token_to_bytes_map.len()
    }

    fn encode(&self, content: &str) -> Result<Vec<u32>> {
        Ok(self.encode_lossy(content))
    }

    /// Note that this isn't actually lossy.
    fn encode_lossy(&self, content: &str) -> Vec<u32> {
        let tokens = content
            .as_bytes()
            .iter()
            .map(|&u8| u8 as u32)
            .collect::<Vec<_>>();

        pair_compress(tokens, &self.pair_to_token_map)
    }

    fn decode(&self, tokens: &Vec<u32>) -> Result<String> {
        let mut result: Vec<u8> = Vec::with_capacity(tokens.len());
        for token in tokens {
            let Some(bytes) = self.token_to_bytes_map.get(token) else {
                return Err(anyhow!("Invalid token: {token}"));
            };
            result.extend(bytes);
        }
        Ok(String::from_utf8(result)?)
    }

    fn into_tensor(self, _device: &candle_core::Device) -> Result<candle_core::Tensor> {
        todo!()
    }
}

pub struct CharPairTokenizer {
    pair_to_token_map: HashMap<(u32, u32), u32>,
    token_to_chars_map: HashMap<u32, Vec<char>>,
    initial_vocab: CharTokenizer,
}

impl CharPairTokenizer {
    pub fn new<T: AsRef<str>>(
        corpus: T,
        initial_vocab: CharTokenizer,
        vocab_size: usize,
    ) -> Result<Self> {
        if vocab_size < initial_vocab.len() {
            return Err(anyhow!(
                "vocab_size must be at least the size of initial_vocab!"
            ));
        }

        let mut tokens = initial_vocab.encode(corpus.as_ref())?;

        let mut curr_vocab_size = initial_vocab.len();
        let mut pair_to_token_map = HashMap::new();
        let mut token_to_chars_map = HashMap::new();

        for i in 0..initial_vocab.len() {
            token_to_chars_map.insert(i as u32, vec![initial_vocab.decode_char(i as u32)?]);
        }

        while curr_vocab_size < vocab_size {
            let pair = get_most_common_pair(&tokens)?;
            let new_token_id = curr_vocab_size as u32;
            curr_vocab_size += 1;
            pair_to_token_map.insert(pair, new_token_id);
            let new_token_chars = token_to_chars_map
                .get(&pair.0)
                .unwrap()
                .iter()
                .chain(token_to_chars_map.get(&pair.1).unwrap().iter())
                .copied()
                .collect::<Vec<_>>();
            token_to_chars_map.insert(new_token_id, new_token_chars);
            tokens = merge(&tokens, pair, new_token_id);
        }

        Ok(Self {
            pair_to_token_map,
            token_to_chars_map,
            initial_vocab,
        })
    }
}

impl Tokenizer for CharPairTokenizer {
    fn len(&self) -> usize {
        self.token_to_chars_map.len()
    }

    fn encode(&self, content: &str) -> Result<Vec<u32>> {
        let tokens = self.initial_vocab.encode(content)?;
        Ok(pair_compress(tokens, &self.pair_to_token_map))
    }

    fn encode_lossy(&self, content: &str) -> Vec<u32> {
        let tokens = self.initial_vocab.encode_lossy(content);
        pair_compress(tokens, &self.pair_to_token_map)
    }

    fn decode(&self, tokens: &Vec<u32>) -> Result<String> {
        let mut result: Vec<char> = Vec::with_capacity(tokens.len());
        for token in tokens {
            let Some(chars) = self.token_to_chars_map.get(token) else {
                return Err(anyhow!("Invalid token: {token}"));
            };
            result.extend(chars);
        }
        Ok(result.into_iter().collect())
    }

    fn into_tensor(self, _device: &candle_core::Device) -> Result<candle_core::Tensor> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        char_tokenizer::CharTokenizer,
        pair_tokenizers::{BytePairTokenizer, CharPairTokenizer, get_most_common_pair, merge},
        tokenizer::Tokenizer,
    };

    #[test]
    fn test_merge() {
        assert_eq!(merge(&[5, 6, 6, 7, 9, 1], (6, 7), 99), vec![5, 6, 99, 9, 1])
    }

    #[test]
    fn test_get_most_common_pair() {
        assert_eq!(
            get_most_common_pair(&[5, 1, 2, 7, 9, 1, 2]).unwrap(),
            (1, 2)
        );
    }

    #[test]
    fn test_get_most_common_pair_resolves_ties_deterministically() {
        assert_eq!(
            get_most_common_pair(&[1, 2, 3, 4, 1, 2, 3, 4]).unwrap(),
            (1, 2)
        );
    }

    #[test]
    fn test_byte_pair_tokenizer() {
        let tokenizer = BytePairTokenizer::new("abcFOOdeFOO", 258).unwrap();
        assert_eq!(tokenizer.encode("a").unwrap(), vec![97]);
        assert_eq!(
            tokenizer.encode("abcFOOdeFOOfFO").unwrap(),
            vec![97, 98, 99, 257, 100, 101, 257, 102, 256]
        );
        assert_eq!(
            tokenizer
                .decode(&vec![97, 98, 99, 257, 100, 101, 257, 102, 256])
                .unwrap(),
            "abcFOOdeFOOfFO".to_owned()
        )
    }

    #[test]
    fn test_char_pair_tokenizer() {
        let tok = CharTokenizer::from_string(&String::from("alo")).unwrap();
        let cptok = CharPairTokenizer::new("alolo", tok, 4).unwrap();
        assert_eq!(cptok.encode("alolo").unwrap(), vec![0, 3, 3]);
        assert_eq!(cptok.decode(&vec![0, 3, 3]).unwrap(), "alolo".to_owned());
    }
}
