use core::f32;

use anyhow::{Result, anyhow};
use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, Sequential, VarBuilder,
    ops::{dropout, softmax},
};

use crate::language_model::LanguageModel;

/// Epsilon for layer norm is what's added to the denominator
/// to make sure it works when the variance is zero. This is just
/// Pytorch's default.
const LAYER_NORM_EPSILON: f64 = 1e-5;

/// Number of output dimensions in the feed-forward layer.
const FEED_FORWARD_OUTPUT_DIMS: usize = 4;

/// This is similar to candle_nn::Dropout with a few salient differences:
///
///   * Instead of implementing `ModuleT`, it detects whether
///     we're training by checking whether the input tensor is part of a
///     computation graph.
///
///   * If `drop_p` is zero, we disable dropout completely, which means
///     there's no efficiency penalty.
struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }
}

impl Module for Dropout {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let is_training = xs.track_op();

        if self.drop_p > 0.0 && is_training {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

/// We're implementing our own layer norm because Candle's built-in one doesn't
/// seem to support backprop: https://github.com/huggingface/candle/issues/2977
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(n_embed: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(n_embed, "weight", candle_nn::init::ONE)?;
        let bias = vb.get_with_hints(n_embed, "bias", candle_nn::init::ZERO)?;
        Ok(LayerNorm { weight, bias })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // This is based on Karpathy's version, described here:
        // https://youtu.be/kCc8FmEb1nY?si=94JxZktfO9lKeJ-K&t=5609
        let xmean = xs.mean_keepdim(D::Minus1)?;
        let xvar = xs.var_keepdim(D::Minus1)?;
        let xhat = xs
            .broadcast_sub(&xmean)?
            .broadcast_div(&(xvar + LAYER_NORM_EPSILON)?.sqrt()?)?;
        let out = self
            .weight
            .broadcast_mul(&xhat)?
            .broadcast_add(&self.bias)?;
        Ok(out)
    }
}

struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
    head_size: usize,
    tril: Tensor,
    neg_infinity: Tensor,
    dropout: Dropout,
}

impl AttentionHead {
    pub fn new(
        n_embed: usize,
        block_size: usize,
        head_size: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let key = candle_nn::linear_no_bias(n_embed, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(n_embed, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(n_embed, head_size, vb.pp("value"))?;
        let tril = Tensor::tril2(block_size, DType::U8, vb.device())?;
        let dropout = Dropout::new(drop_p);

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
            dropout,
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
        let wei = self.dropout.forward(&wei)?;
        let out = wei.matmul(&v)?;
        Ok(out)
    }
}

struct MultiAttentionHead {
    heads: Vec<AttentionHead>,
    proj: Linear,
    dropout: Dropout,
}

impl MultiAttentionHead {
    pub fn new(
        n_embed: usize,
        block_size: usize,
        num_heads: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            heads.push(AttentionHead::new(
                n_embed,
                block_size,
                n_embed / num_heads,
                drop_p,
                vb.pp(format!("head_{i}")),
            )?);
        }
        let proj = candle_nn::linear(n_embed, n_embed, vb.pp("proj"))?;
        let dropout = Dropout::new(drop_p);
        Ok(Self {
            heads,
            proj,
            dropout,
        })
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
        let cat = Tensor::cat(&heads, 2)?;
        let out = self.proj.forward(&cat)?;
        let out = self.dropout.forward(&out)?;
        Ok(out)
    }
}

pub struct FeedForward {
    ff: Linear,
    proj: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(n_embed: usize, drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let ff = candle_nn::linear(n_embed, FEED_FORWARD_OUTPUT_DIMS * n_embed, vb.pp("ff"))?;
        let proj = candle_nn::linear(FEED_FORWARD_OUTPUT_DIMS * n_embed, n_embed, vb.pp("proj"))?;
        let dropout = Dropout::new(drop_p);
        Ok(Self { ff, proj, dropout })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let out = self.ff.forward(xs)?.relu()?;
        let out = self.proj.forward(&out)?;
        let out = self.dropout.forward(&out)?;
        Ok(out)
    }
}

pub struct Block {
    sa_heads: MultiAttentionHead,
    feed_forward: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    pub fn new(
        n_embed: usize,
        block_size: usize,
        num_heads: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sa_heads =
            MultiAttentionHead::new(n_embed, block_size, num_heads, drop_p, vb.pp("sa_heads"))?;
        let ln1 = LayerNorm::new(n_embed, vb.pp("layer_norm1"))?;
        let feed_forward = FeedForward::new(n_embed, drop_p, vb.pp("feed_forward"))?;
        let ln2 = LayerNorm::new(n_embed, vb.pp("layer_norm2"))?;
        Ok(Self {
            sa_heads,
            feed_forward,
            ln1,
            ln2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = (xs + self.sa_heads.forward(&self.ln1.forward(&xs)?)?)?;
        let xs = (&xs + self.feed_forward.forward(&self.ln2.forward(&xs)?)?)?;
        Ok(xs)
    }
}

pub struct TransformerLanguageModel {
    block_size: usize,
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    positions: Tensor,
    blocks: Sequential,
    layer_norm: LayerNorm,
    language_head: Linear,
}

#[derive(Copy, Clone)]
pub struct TransformerLanguageModelOptions {
    pub n_embed: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub drop_p: f32,
}

impl TransformerLanguageModel {
    pub fn new(options: TransformerLanguageModelOptions, vb: VarBuilder) -> Result<Self> {
        let TransformerLanguageModelOptions {
            n_embed,
            block_size,
            num_layers,
            num_heads,
            vocab_size,
            drop_p,
        } = options;

        if num_heads <= 0 {
            return Err(anyhow!("must have a positive number of attention heads!"));
        }
        if n_embed as f64 / num_heads as f64 != (n_embed / num_heads) as f64 {
            return Err(anyhow!(
                "embedding dimensions ({n_embed}) must be divisible by attention heads ({num_heads})!",
            ));
        }

        let device = vb.device();
        let token_embedding_table =
            candle_nn::embedding(vocab_size, n_embed, vb.pp("token_embedding_table"))?;
        let position_embedding_table =
            candle_nn::embedding(block_size, n_embed, vb.pp("position_embedding_table"))?;
        let positions = Tensor::arange(0 as u32, block_size as u32, device)?;
        let mut blocks = candle_nn::seq();
        for i in 0..num_layers {
            blocks = blocks.add(Block::new(
                n_embed,
                block_size,
                num_heads,
                drop_p,
                vb.pp(format!("block{i}")),
            )?);
        }
        let layer_norm = LayerNorm::new(n_embed, vb.pp("layer_norm"))?;
        let language_head = candle_nn::linear(n_embed, vocab_size, vb.pp("language_head"))?;
        Ok(Self {
            block_size,
            token_embedding_table,
            position_embedding_table,
            positions,
            blocks,
            layer_norm,
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
        let out = self.blocks.forward(&x)?;
        let out = self.layer_norm.forward(&out)?;
        let logits = self.language_head.forward(&out)?;
        Ok(logits)
    }
}

impl LanguageModel for TransformerLanguageModel {
    fn block_size(&self) -> usize {
        self.block_size
    }
}
