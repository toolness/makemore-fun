use core::f32;

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, ops::softmax};

use crate::BLOCK_SIZE;

/// Number of dimensions in embedding space.
const N_EMBED: usize = 32;

/// Number of dimensions in self-attention heads
const HEAD_SIZE: usize = 16;

pub struct TransformerLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    positions: Tensor,
    key: Linear,
    query: Linear,
    language_head: Linear,
}

impl TransformerLanguageModel {
    pub fn new(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let device = vb.device();
        let token_embedding_table =
            candle_nn::embedding(vocab_size, N_EMBED, vb.pp("token_embedding_table"))?;
        let position_embedding_table =
            candle_nn::embedding(BLOCK_SIZE, N_EMBED, vb.pp("position_embedding_table"))?;
        let positions = Tensor::arange(0 as u32, BLOCK_SIZE as u32, device)?;
        let key = candle_nn::linear_no_bias(N_EMBED, HEAD_SIZE, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(N_EMBED, HEAD_SIZE, vb.pp("query"))?;
        let language_head = candle_nn::linear(N_EMBED, vocab_size, vb.pp("language_head"))?;
        // TODO: Add the rest of the modules!
        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            positions,
            key,
            query,
            language_head,
        })
    }
}

impl Module for TransformerLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        let device = self.positions.device();
        let time_steps = xs.dims2()?.1;
        let tok_emb = self.token_embedding_table.forward(xs)?;
        let pos_emb = self
            .position_embedding_table
            .forward(&self.positions.i(0..time_steps)?)?;
        let x = tok_emb.broadcast_add(&pos_emb)?;
        let k = self.key.forward(&x)?;
        let q = self.query.forward(&x)?;

        // TODO: wei shouldn't be zero, it should be q @ k.T
        let _ = (k, q);
        let wei = Tensor::zeros((time_steps, time_steps), DType::F32, device)?;
        let tril_mask = Tensor::tril2(time_steps, DType::U8, device)?;
        let wei = tril_mask.where_cond(
            &wei,
            &Tensor::full(f32::NEG_INFINITY, (time_steps, time_steps), device)?,
        )?;
        let wei = softmax(&wei, 1)?;
        // TODO: Finish implementing this.
        println!("{}", wei);

        let logits = self.language_head.forward(&x)?;
        Ok(logits)
    }
}
