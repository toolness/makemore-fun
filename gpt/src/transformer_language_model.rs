use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Linear, Module, VarBuilder};

/// Number of dimensions in embedding space.
const N_EMBED: usize = 32;

pub struct TransformerLanguageModel {
    token_embedding_table: Embedding,
    language_head: Linear,
}

impl TransformerLanguageModel {
    pub fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(vocab_size, N_EMBED, vb.pp("embedding"))?;
        let language_head = candle_nn::linear_no_bias(N_EMBED, vocab_size, vb.pp("language_head"))?;
        // TODO: Add the rest of the modules!
        Ok(Self {
            token_embedding_table,
            language_head,
        })
    }
}

impl Module for TransformerLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        let xs = self.token_embedding_table.forward(xs)?;
        let xs = self.language_head.forward(&xs)?;
        // TODO: Implement the rest of this!
        Ok(xs)
    }
}
