use std::io::{self, Write};

use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{Module, loss::cross_entropy, ops::softmax};
use rand::rngs::StdRng;

use crate::{
    tokenizer::Tokenizer,
    util::{assert_equal_tensors, multinomial},
};

pub fn language_loss(logits: &Tensor, ys: &Tensor) -> Result<Tensor> {
    let (batch_size, block_size, vocab_size) = logits.dims3()?;

    if cfg!(debug_assertions) {
        let sm = softmax(&logits, D::Minus1)?;
        assert_relative_eq!(
            sm.get(0)?.get(0)?.sum(0)?.to_scalar::<f32>()?,
            1.0,
            epsilon = 0.0001
        );
    }

    let flat_logits = logits.reshape((batch_size * block_size, vocab_size))?;
    let flat_ys = ys.reshape(batch_size * block_size)?;

    if cfg!(debug_assertions) {
        assert_equal_tensors(logits.get(0)?.get(0)?, flat_logits.get(0)?)?;
    }
    let loss = cross_entropy(&flat_logits, &flat_ys)?;
    Ok(loss)
}

pub fn language_generate_and_print(
    context: &Vec<u32>,
    temperature: f32,
    model: &Box<dyn Module>,
    block_size: usize,
    num_chars: usize,
    rng: &mut StdRng,
    device: &Device,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>> {
    let mut result = Vec::with_capacity(context.len() + num_chars);
    result.extend(context.iter());
    for _ in 0..num_chars {
        let block_slice = &result[result.len().saturating_sub(block_size)..];
        let block = Tensor::from_slice(block_slice, (1, block_slice.len()), device)?;
        let logits = model.forward(&block)?;
        // Take just the logits for the final time step.
        let logits = logits.i((.., block_slice.len() - 1, ..))?;
        let token = if temperature == 0.0 {
            logits.argmax(1)?.get(0)?.to_scalar()?
        } else {
            // It's very weird that I can't just use the division operator here.
            let logits =
                logits.broadcast_div(&Tensor::from_slice(&[temperature], (1,), device)?)?;
            let sm = softmax(&logits, 1)?;
            multinomial(&sm, rng)?
        };
        print!("{}", tokenizer.decode(&vec![token])?);
        io::stdout().flush()?;
        result.push(token);
    }
    println!("");
    Ok(result)
}
