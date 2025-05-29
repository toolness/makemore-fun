use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::Module;
use rand::rngs::StdRng;

pub trait LanguageModel: Module {
    fn loss(&self, logits: &Tensor, ys: &Tensor) -> Result<Tensor>;

    fn generate(&self, num_chars: usize, rng: &mut StdRng, device: &Device) -> Result<Vec<u32>>;
}
