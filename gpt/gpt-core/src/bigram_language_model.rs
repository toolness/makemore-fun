use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Module, VarBuilder};

pub struct BigramLanguageModel {
    token_embedding_table: Embedding,
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, vocab_size, vb)?;
        Ok(Self {
            token_embedding_table,
        })
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        Ok(self.token_embedding_table.forward(xs)?)
    }
}
