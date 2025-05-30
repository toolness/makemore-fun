use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{Module, loss::cross_entropy, ops::softmax};
use rand::rngs::StdRng;

use crate::{
    BATCH_SIZE, BLOCK_SIZE,
    util::{assert_equal_tensors, multinomial},
};

pub fn language_loss(logits: &Tensor, ys: &Tensor) -> Result<Tensor> {
    let vocab_size = logits.dims3()?.2;
    assert_eq!(logits.dims3()?, (BATCH_SIZE, BLOCK_SIZE, vocab_size));

    if cfg!(debug_assertions) {
        let sm = softmax(&logits, D::Minus1)?;
        assert_relative_eq!(
            sm.get(0)?.get(0)?.sum(0)?.to_scalar::<f32>()?,
            1.0,
            epsilon = 0.0001
        );
    }

    let flat_logits = logits.reshape((BATCH_SIZE * BLOCK_SIZE, vocab_size))?;
    let flat_ys = ys.reshape(BATCH_SIZE * BLOCK_SIZE)?;

    if cfg!(debug_assertions) {
        assert_equal_tensors(logits.get(0)?.get(0)?, flat_logits.get(0)?)?;
    }
    let loss = cross_entropy(&flat_logits, &flat_ys)?;
    Ok(loss)
}

pub fn language_generate(
    model: &Box<dyn Module>,
    num_chars: usize,
    rng: &mut StdRng,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut result = Vec::with_capacity(num_chars);
    result.push(0);
    for _ in 0..num_chars {
        let block_slice = &result[result.len().saturating_sub(BLOCK_SIZE)..];
        let block = Tensor::from_slice(block_slice, (1, block_slice.len()), device)?;
        let logits = model.forward(&block)?;
        // Take just the logits for the final time step.
        let logits = logits.i((.., block_slice.len() - 1, ..))?;
        let sm = softmax(&logits, 1)?;
        let token = multinomial(&sm, rng)?;
        result.push(token);
    }
    Ok(result)
}
