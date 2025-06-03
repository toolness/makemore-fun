use std::collections::HashMap;

use crate::bigram_language_model::BigramLanguageModel;
use crate::device::Device;
use crate::transformer_language_model::TransformerLanguageModel;
use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
pub enum Model {
    Bigram,
    Transformer,
}

#[derive(Parser)]
pub struct Args {
    /// Whether to display information about the variables in the network.
    #[arg(long, default_value_t = false)]
    pub vars: bool,

    /// Random number seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of epochs to train the model.
    #[arg(long, default_value_t = 3_000)]
    pub epochs: usize,

    /// Context window size, measured in characters.
    #[arg(long, default_value_t = 8)]
    pub block_size: usize,

    /// Number of characters of output to generate.
    #[arg(long, default_value_t = 500)]
    pub chars: usize,

    /// The text file to use as a training corpus.
    #[arg(long, default_value_t = String::from("tiny-shakespeare.txt"))]
    pub corpus: String,

    /// The file to save the trained weights to, in safetensors format.
    #[arg(long)]
    pub save: Option<String>,

    /// The file to load the trained weights from, in safetensors format.
    #[arg(long)]
    pub load: Option<String>,

    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    pub device: Device,

    #[arg(long, value_enum, default_value_t = Model::Transformer)]
    pub model: Model,

    /// Number of training examples per batch.
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// Number of self-attention/feed-forward layers (transformer model only).
    #[arg(long, default_value_t = 1)]
    pub layers: usize,

    /// Number of attention heads per layer (transformer model only).
    #[arg(long, default_value_t = 4)]
    pub heads: usize,

    /// Number of dimensions in embedding space (transformer model only).
    #[arg(long, default_value_t = 32)]
    pub embedding_dims: usize,

    /// Dropout probability (transformer model only).
    #[arg(long, default_value_t = 0.0)]
    pub dropout: f32,

    /// The learning rate.
    #[arg(long, default_value_t = 0.01)]
    pub lr: f64,

    /// Initial context to pass into the model when generating content.
    #[arg(long, default_value_t = String::from("\n"))]
    pub context: String,
}

impl Args {
    pub fn create_model(&self, vocab_size: usize, vb: VarBuilder) -> Result<Box<dyn Module>> {
        match self.model {
            Model::Bigram => Ok(Box::new(BigramLanguageModel::new(vocab_size, vb)?)),
            Model::Transformer => Ok(Box::new(TransformerLanguageModel::new(
                self.embedding_dims,
                self.block_size,
                self.layers,
                self.heads,
                vocab_size,
                self.dropout,
                vb,
            )?)),
        }
    }

    pub fn create_model_no_grad(
        &self,
        vocab_size: usize,
        varmap: &VarMap,
        device: &candle_core::Device,
    ) -> Result<Box<dyn Module>> {
        // "Freeze" the varmap as detached tensors to ensure that gradients aren't calculated
        // for our parameters. While this doesn't actually seem to improve performance, it _does_
        // seem to result in better training, since our evals don't mess with our optimizer: when
        // running with `--epochs=5000 --lr=1e-3 --blocks=1` the loss improves from 2.193 to 2.179
        // when using the no-gradient variant of the model for evals.
        let varmap_data = varmap.data().lock().unwrap();
        let mut detached_vars: HashMap<String, Tensor> = HashMap::with_capacity(varmap_data.len());
        for (path, var) in varmap_data.iter() {
            detached_vars.insert(path.clone(), var.as_detached_tensor());
        }
        self.create_model(
            vocab_size,
            VarBuilder::from_tensors(detached_vars, DType::F32, device),
        )
    }
}
