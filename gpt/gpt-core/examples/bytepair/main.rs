use std::{collections::HashMap, iter::zip};

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
    let new_bytes = merge(&bytes, most_common_pair, vocab_size);

    println!(
        "Incorporated new token into vocabulary, length of new bytes is {}.",
        new_bytes.len()
    );
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

#[cfg(test)]
mod tests {
    use crate::merge;

    #[test]
    fn test_merge() {
        assert_eq!(merge(&[5, 6, 6, 7, 9, 1], (6, 7), 99), vec![5, 6, 99, 9, 1])
    }
}
