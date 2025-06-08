use std::{cmp::Ordering, collections::HashMap, iter::zip};

use anyhow::{Result, anyhow};

/// This is the first paragraph from
/// https://www.reedbeta.com/blog/programmers-intro-to-unicode/
const UNICODE_STR: &'static str = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.";

/// This is the exercise from the beginning of Karpathy's
/// "Let's build the GPT Tokenizer" lecture at:
///
///   https://www.youtube.com/watch?v=zduSFxRajkE
pub fn main() {
    let unicode_len = UNICODE_STR.chars().count();
    let bytes = UNICODE_STR
        .as_bytes()
        .iter()
        .map(|&u8| u8 as u32)
        .collect::<Vec<_>>();
    let mut vocab_size = 255;
    println!(
        "Number of unicode chars: {unicode_len}\nNumber of UTF-8 bytes: {}",
        bytes.len()
    );

    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    for (&a, &b) in zip(&bytes, &bytes[1..]) {
        let bytepair = (a, b);
        let entry = counts.entry(bytepair).or_insert(0);
        *entry += 1;
    }

    let mut all = counts.into_iter().collect::<Vec<_>>();
    all.sort_by(|(_, a_count), (_, b_count)| b_count.cmp(a_count));

    let (most_common_pair, count) = all[0];
    let (a, b) = most_common_pair;
    println!(
        "Most common pair with {count} occurrences: {a} ({:?}) {b} ({:?})",
        char::from_u32(a as u32),
        char::from_u32(b as u32)
    );
    vocab_size += 1;
    let new_tokens = merge(&bytes, most_common_pair, vocab_size);

    println!(
        "Incorporated new token into vocabulary, length of new bytes is {}.",
        new_tokens.len()
    );

    let tokenizer = BytePairTokenizer::new(UNICODE_STR, 257).unwrap();
    assert_eq!(tokenizer.encode(UNICODE_STR), new_tokens);
    println!("BytePairTokenizer::encode() works!");
    assert_eq!(
        tokenizer.decode(&new_tokens).unwrap(),
        UNICODE_STR.to_owned()
    );
    println!("BytePairTokenizer::decode() works!");
}

fn merge(tokens: &[u32], pair: (u32, u32), pair_token_id: u32) -> Vec<u32> {
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

struct BytePairTokenizer {
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

    pub fn encode<T: AsRef<str>>(&self, string: T) -> Vec<u32> {
        // This is pretty inefficient and can probably be improved a lot.
        let mut tokens = string
            .as_ref()
            .as_bytes()
            .iter()
            .map(|&u8| u8 as u32)
            .collect::<Vec<_>>();

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
                    if let Some(&token) = self.pair_to_token_map.get(&curr_pair) {
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

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result: Vec<u8> = Vec::with_capacity(tokens.len());
        for token in tokens {
            let Some(bytes) = self.token_to_bytes_map.get(token) else {
                return Err(anyhow!("Invalid token: {token}"));
            };
            result.extend(bytes);
        }
        Ok(String::from_utf8(result)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{BytePairTokenizer, get_most_common_pair, merge};

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
    fn test_tokenizer() {
        let tokenizer = BytePairTokenizer::new("abcFOOdeFOO", 258).unwrap();
        assert_eq!(
            tokenizer.encode("abcFOOdeFOOfFO"),
            vec![97, 98, 99, 257, 100, 101, 257, 102, 256]
        );
    }
}
