use crate::device::Device;
use anyhow::Result;
use candle_core::Tensor;
use clap::{Parser, ValueEnum};
use gpt_core::char_tokenizer::CharTokenizer;
use gpt_core::language_model_builder::LanguageModelBuilder;
use gpt_core::pair_tokenizers::{CharPairFilter, CharPairTokenizer};
use gpt_core::tokenizer::Tokenizer;
use gpt_core::transformer_language_model::TransformerLanguageModelOptions;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone, ValueEnum)]
pub enum Model {
    /// A simple bigram model, similar to a Markov chain.
    Bigram,

    /// A transformer model, similar to the one documented in the
    /// "Attention is all you need" paper.
    Transformer,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum ArgsTokenizerType {
    /// Each character from the training corpus is a separate token.
    Char,

    /// Each character from the training corpus is a separate token, but
    /// we also mint frequent occurrences of consecutive alphabetic characters
    /// as new tokens.
    CharPairAlpha,
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

    /// The device to run training and inference on.
    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    pub device: Device,

    /// The neural net model architecture to use for training/inference.
    #[arg(long, value_enum, default_value_t = Model::Transformer)]
    pub model: Model,

    /// The tokenizer to use for encoding/decoding content that is
    /// fed into/retrieved from the model.
    #[arg(long, value_enum, default_value_t = ArgsTokenizerType::Char)]
    pub tokenizer: ArgsTokenizerType,

    /// Size of tokenizer vocabulary (not used by char tokenizer).
    #[arg(long, default_value_t = 300)]
    pub vocab_size: usize,

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

    pub fn create_tokenizer_and_training_data(
        &self,
        device: &candle_core::Device,
        existing_tokenizer: Option<Box<dyn Tokenizer>>,
    ) -> Result<(Box<dyn Tokenizer>, Tensor)> {
        let training_corpus = std::fs::read_to_string(&self.corpus)?;
        let original_len = training_corpus.len();
        let tokenizer: Box<dyn Tokenizer> = if let Some(tokenizer) = existing_tokenizer {
            // We don't want to re-initialize the tokenizer if we were already passed one in, especially
            // because generating the tokenizer's vocabulary could take a while.
            tokenizer
        } else {
            match self.tokenizer {
                ArgsTokenizerType::Char => Box::new(CharTokenizer::from_string(&training_corpus)?),
                ArgsTokenizerType::CharPairAlpha => {
                    let pb = ProgressBar::new(self.vocab_size as u64);
                    pb.set_style(
                        ProgressStyle::with_template(
                            "[{elapsed_precise}] {bar:30.white/white} {pos:>7}/{len:7} {msg}",
                        )
                        .unwrap(),
                    );
                    pb.set_message("Creating vocab");
                    let initial_vocab = CharTokenizer::from_string(&training_corpus)?;
                    let result = Box::new(CharPairTokenizer::new(
                        &training_corpus,
                        initial_vocab,
                        self.vocab_size,
                        Some(CharPairFilter::AlphaOnly),
                        |progress| {
                            pb.set_position(progress as u64);
                        },
                    )?);
                    pb.finish();
                    result
                }
            }
        };
        let tokens = tokenizer.encode(&training_corpus)?;
        let tokens_len = tokens.len();
        let data = Tensor::new(tokens, device)?;
        println!(
            "Tokenizer training data compression ratio: {:.2}",
            original_len as f32 / tokens_len as f32
        );
        Ok((tokenizer, data))
    }
}
