use anyhow::Result;
use approx::assert_relative_eq;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{Module, loss::cross_entropy, ops::softmax};
use rand::rngs::StdRng;

use crate::{
    tokenizer::Tokenizer,
    util::{assert_equal_tensors, multinomial},
};

pub trait LanguageModel: Module {
    fn block_size(&self) -> usize;
}

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

pub struct LanguageGenerator {
    context: Vec<u32>,
    model: Box<dyn LanguageModel>,
    block_size: usize,
}

impl LanguageGenerator {
    pub fn new(context: &[u32], model: Box<dyn LanguageModel>, block_size: usize) -> Result<Self> {
        let mut context = context[context.len().saturating_sub(block_size)..].to_vec();
        if context.len() == 0 {
            // We need to have *something* to predict the next token, so
            // just add the first token in the vocabulary.
            context.push(0);
        }
        Ok(Self {
            context,
            model,
            block_size,
        })
    }

    pub fn logits(&self, device: &Device) -> Result<Tensor> {
        let block = Tensor::from_slice(&self.context, (1, self.context.len()), device)?;
        let logits = self.model.forward(&block)?;
        // Take just the logits for the final time step.
        let logits = logits.i((.., self.context.len() - 1, ..))?;
        Ok(logits)
    }

    fn push(&mut self, token: u32) {
        if self.context.len() == self.block_size {
            self.context.remove(0);
        }
        self.context.push(token);
    }

    pub fn next_token(
        &mut self,
        rng: &mut StdRng,
        tokenizer: &Box<dyn Tokenizer>,
        temperature: f32,
        device: &Device,
    ) -> Result<String> {
        let logits = self.logits(device)?;
        let token = if temperature == 0.0 {
            logits.argmax(1)?.get(0)?.to_scalar()?
        } else {
            // It's very weird that I can't just use the division operator here.
            let logits =
                logits.broadcast_div(&Tensor::from_slice(&[temperature], (1,), device)?)?;
            let sm = softmax(&logits, 1)?;
            multinomial(&sm, rng)?
        };
        self.push(token);
        Ok(tokenizer.decode(&vec![token])?)
    }
}
