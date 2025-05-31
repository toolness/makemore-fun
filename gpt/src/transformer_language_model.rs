use core::f32;

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, ops::softmax};

use crate::BLOCK_SIZE;

/// Number of dimensions in embedding space.
const N_EMBED: usize = 32;

struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
    head_size: usize,
}

impl AttentionHead {
    pub fn new(head_size: usize, vb: VarBuilder) -> Result<Self> {
        let key = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("value"))?;
        Ok(Self {
            key,
            query,
            value,
            head_size,
        })
    }
}

impl Module for AttentionHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let device = self.key.weight().device();
        let batches = xs.dims3()?.0;
        let time_steps = xs.dims3()?.1;

        let k = self.key.forward(&xs)?;
        let q = self.query.forward(&xs)?;
        let v = self.value.forward(&xs)?;

        let wei = (q.matmul(&k.transpose(1, 2)?)? / (self.head_size as f64).powf(0.5))?;
        assert_eq!(wei.dims3()?, (batches, time_steps, time_steps));
        let tril_mask = Tensor::tril2(time_steps, DType::U8, device)?
            .broadcast_as((batches, time_steps, time_steps))?;
        let wei = tril_mask.where_cond(
            &wei,
            &Tensor::full(f32::NEG_INFINITY, (batches, time_steps, time_steps), device)?,
        )?;
        let wei = softmax(&wei, 2)?;
        let out = wei.matmul(&v)?;
        Ok(out)
    }
}

pub struct TransformerLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    positions: Tensor,
    sa_head: AttentionHead,
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
        let sa_head = AttentionHead::new(N_EMBED, vb.pp("sa_head"))?;
        let language_head = candle_nn::linear(N_EMBED, vocab_size, vb.pp("language_head"))?;
        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            positions,
            sa_head,
            language_head,
        })
    }
}

impl Module for TransformerLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        let time_steps = xs.dims2()?.1;
        let tok_emb = self.token_embedding_table.forward(xs)?;
        let pos_emb = self
            .position_embedding_table
            .forward(&self.positions.i(0..time_steps)?)?;
        let x = tok_emb.broadcast_add(&pos_emb)?;
        let out = self.sa_head.forward(&x)?;
        let logits = self.language_head.forward(&out)?;
        Ok(logits)
    }
}
