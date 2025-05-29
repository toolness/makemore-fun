use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, loss::cross_entropy, ops::softmax};
use rand::rngs::StdRng;

use crate::{
    BATCH_SIZE, BLOCK_SIZE,
    language_model::LanguageModel,
    util::{assert_equal_tensors, multinomial},
};

pub struct BigramLanguageModel {
    vocab_size: usize,
    token_embedding_table: Embedding,
}

impl BigramLanguageModel {
    pub fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, vocab_size, vb)?;
        Ok(Self {
            vocab_size,
            token_embedding_table,
        })
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        Ok(self.token_embedding_table.forward(xs)?)
    }
}

impl LanguageModel for BigramLanguageModel {
    fn loss(&self, logits: &Tensor, ys: &Tensor) -> Result<Tensor> {
        assert_eq!(logits.dims3()?, (BATCH_SIZE, BLOCK_SIZE, self.vocab_size));

        if cfg!(debug_assertions) {
            let sm = softmax(&logits, D::Minus1)?;
            assert_relative_eq!(
                sm.get(0)?.get(0)?.sum(0)?.to_scalar::<f32>()?,
                1.0,
                epsilon = 0.0001
            );
        }

        let flat_logits = logits.reshape((BATCH_SIZE * BLOCK_SIZE, self.vocab_size))?;
        let flat_ys = ys.reshape(BATCH_SIZE * BLOCK_SIZE)?;

        if cfg!(debug_assertions) {
            assert_equal_tensors(logits.get(0)?.get(0)?, flat_logits.get(0)?)?;
        }
        let loss = cross_entropy(&flat_logits, &flat_ys)?;
        Ok(loss)
    }

    fn generate(&self, num_chars: usize, rng: &mut StdRng, device: &Device) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(num_chars);
        result.push(0);
        for _ in 0..num_chars {
            let block_slice = &result[result.len().saturating_sub(BLOCK_SIZE)..];
            let block = Tensor::from_slice(block_slice, (1, block_slice.len()), device)?;
            let logits = self.forward(&block)?;
            // Take just the logits for the final time step.
            let logits = logits.i((.., block_slice.len() - 1, ..))?;
            let sm = softmax(&logits, 1)?;
            let token = multinomial(&sm, rng)?;
            result.push(token);
        }
        Ok(result)
    }
}
