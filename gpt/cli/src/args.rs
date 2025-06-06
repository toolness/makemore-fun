use crate::device::Device;
use clap::{Parser, ValueEnum};
use gpt_core::language_model_builder::LanguageModelBuilder;
use gpt_core::transformer_language_model::TransformerLanguageModelOptions;

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

    /// Temperature to use when generating content.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,

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
    pub fn language_model_builder(&self, vocab_size: usize) -> LanguageModelBuilder {
        match self.model {
            Model::Bigram => LanguageModelBuilder::Bigram(vocab_size),
            Model::Transformer => {
                LanguageModelBuilder::Transformer(TransformerLanguageModelOptions {
                    n_embed: self.embedding_dims,
                    block_size: self.block_size,
                    num_layers: self.layers,
                    num_heads: self.heads,
                    vocab_size,
                    drop_p: self.dropout,
                })
            }
        }
    }
}
