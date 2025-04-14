use std::collections::{HashMap, HashSet};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};

/// Loads names.txt and splits it by newlines
fn get_names_txt() -> Result<Vec<String>> {
    let content = std::fs::read_to_string("names.txt")?;
    let lines = content.lines().map(String::from).collect();
    Ok(lines)
}

struct Bigrams {
    ctoi: HashMap<char, usize>,
    itoc: HashMap<usize, char>,
    tensor: Tensor
}

impl Bigrams {
    pub fn from_names(names: &Vec<String>, device: &Device) -> Result<Self> {
        let mut all_chars: HashSet<char> = HashSet::new();
        for name in names {
            all_chars.extend(name.chars());
        }
        let mut all_chars_sorted: Vec<char> = all_chars.iter().copied().collect();
        all_chars_sorted.sort();
        let num_chars = all_chars_sorted.len() + 1;
        let tensor = Tensor::zeros((num_chars, num_chars), DType::F32, &device)?;
        let mut ctoi: HashMap<char, usize> = HashMap::new();
        let mut itoc = HashMap::new();
        ctoi.insert('.', 0);
        itoc.insert(0, '.');
        for (i, char, ) in all_chars_sorted.iter().enumerate() {
            ctoi.insert(*char, i + 1);
            itoc.insert(i + 1, *char);
        }
    
        let mut result = Bigrams {
            ctoi,
            itoc,
            tensor
        };

        result.populate(names)?;

        Ok(result)
    }

    fn populate(&mut self, names: &Vec<String>) -> Result<()> {
        for name in names {
            let first_char = name.chars().nth(0).unwrap();
            self.increment(('.', first_char))?;
            for bigram in name.chars().zip(name.chars().skip(1)) {
                self.increment(bigram)?;
            }
            let last_char = name.chars().last().unwrap();
            self.increment(('.', last_char))?;
        }
        Ok(())
    }

    fn increment(&mut self, (first, second): (char, char)) -> Result<()> {
        let first_idx = *self.ctoi.get(&first).unwrap();
        let second_idx = *self.ctoi.get(&second).unwrap();
        let value = self.tensor.get(first_idx)?.get(second_idx)?;
        let orig: f32 = value.to_scalar()?;
        let var = Var::from_tensor(&value)?;

        // TODO: This doesn't work well b/c tensors in candle are immutable, unlike in torch.
        let inc_by_one = Tensor::from_slice(&[orig + 1.0], (), &self.tensor.device())?;
        println!("BLAH {:?} INC BY ONE {:?}", value, inc_by_one);
        var.set(&inc_by_one)?;
        Ok(())
        //self.tensor.index_add([0].into(), &[1], 0);
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let names = get_names_txt()?;

    println!("total names: {}", names.len());
    println!("first 10 names: {:?}", &names[..10]);
    println!("min name len: {}", names.iter().map(|name| name.len()).min().unwrap());
    println!("max name len: {}", names.iter().map(|name| name.len()).max().unwrap());

    let bigrams = Bigrams::from_names(&names, &device)?;

    println!("HELLO {:?}", bigrams.tensor.get(1)?.get(1)?);

    //println!("bigram 'aa' count: {:?}", bigrams.raw.get(&('a', 'a')));

    Ok(())
}
