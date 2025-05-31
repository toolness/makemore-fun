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
    tril: Tensor,
    neg_infinity: Tensor,
}

impl AttentionHead {
    pub fn new(head_size: usize, vb: VarBuilder) -> Result<Self> {
        let key = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(N_EMBED, head_size, vb.pp("value"))?;
        let tril = Tensor::tril2(BLOCK_SIZE, DType::U8, vb.device())?;

        // It's annoying that we have to put this in its own tensor, since it's just
        // a matrix full of negative infinity... I wish we could just use a scalar or something.
        let neg_infinity = Tensor::full(f32::NEG_INFINITY, (1,), vb.device())?;

        Ok(Self {
            key,
            query,
            value,
            head_size,
            tril,
            neg_infinity,
        })
    }
}

impl Module for AttentionHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batches, time_steps, _) = xs.dims3()?;

        let k = self.key.forward(&xs)?;
        let q = self.query.forward(&xs)?;
        let v = self.value.forward(&xs)?;

        let wei = (q.matmul(&k.transpose(1, 2)?)? / (self.head_size as f64).powf(0.5))?;
        assert_eq!(wei.dims3()?, (batches, time_steps, time_steps));
        let tril_mask = self
            .tril
            .i((0..time_steps, 0..time_steps))?
            .broadcast_as((batches, time_steps, time_steps))?;
        let wei = tril_mask.where_cond(
            &wei,
            &self
                .neg_infinity
                .broadcast_as((batches, time_steps, time_steps))?,
        )?;
        let wei = softmax(&wei, 2)?;
        let out = wei.matmul(&v)?;
        Ok(out)
    }
}

struct MultiAttentionHead {
    heads: Vec<AttentionHead>,
}

impl MultiAttentionHead {
    pub fn new(num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let mut heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            heads.push(AttentionHead::new(
                N_EMBED / num_heads,
                vb.pp(format!("head_{i}")),
            )?);
        }
        Ok(Self { heads })
    }
}

impl Module for MultiAttentionHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut heads = Vec::with_capacity(self.heads.len());
        // TODO: I think these heads are going to be executed serially, rather than in parallel.
        // But I'm just following Karpathy's video for now, and he does this using a Python
        // list comprehension, so I'm just gonna do what he's doing for now.
        for head in self.heads.iter() {
            heads.push(head.forward(xs)?);
        }
        Ok(Tensor::cat(&heads, 2)?)
    }
}

pub struct Block {
    sa_heads: MultiAttentionHead,
    feed_forward: Linear,
}

impl Block {
    pub fn new(num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let sa_heads = MultiAttentionHead::new(num_heads, vb.pp("sa_heads"))?;
        let feed_forward = candle_nn::linear(N_EMBED, N_EMBED, vb.pp("feed_forward"))?;
        Ok(Self {
            sa_heads,
            feed_forward,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let out = self.sa_heads.forward(&xs)?;
        let out = self.feed_forward.forward(&out)?.relu()?;
        Ok(out)
    }
}

pub struct TransformerLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    positions: Tensor,
    block: Block,
    language_head: Linear,
}

impl TransformerLanguageModel {
    pub fn new(vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let token_embedding_table =
            candle_nn::embedding(vocab_size, N_EMBED, vb.pp("token_embedding_table"))?;
        let position_embedding_table =
            candle_nn::embedding(BLOCK_SIZE, N_EMBED, vb.pp("position_embedding_table"))?;
        let positions = Tensor::arange(0 as u32, BLOCK_SIZE as u32, device)?;
        let block = Block::new(4, vb.pp("block"))?;
        let language_head = candle_nn::linear(N_EMBED, vocab_size, vb.pp("language_head"))?;
        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            positions,
            block,
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
        let out = self.block.forward(&x)?;
        let logits = self.language_head.forward(&out)?;
        Ok(logits)
    }
}
