use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{Module, VarBuilder, VarMap};
use gpt_core::{
    language_model::LanguageGenerator,
    language_model_builder::LanguageModelBuilder,
    tokenizer::{TOKENIZER_VOCABULARY_KEY, Tokenizer},
    transformer_language_model::{TransformerLanguageModel, TransformerLanguageModelOptions},
    util::load_data_from_safetensors,
};
use rand::{SeedableRng, rngs::StdRng};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LanguageModel {
    module: Arc<Box<dyn Module>>,
    tokenizer: Arc<Tokenizer>,
}

#[wasm_bindgen]
impl LanguageModel {
    #[wasm_bindgen]
    pub fn bigram(safetensors_u8: &[u8]) -> Result<Self, JsError> {
        Self::load_safetensors_and_build(safetensors_u8, |vocab_size| {
            LanguageModelBuilder::Bigram(vocab_size)
        })
    }

    #[wasm_bindgen]
    pub fn transformer(
        n_embed: usize,
        block_size: usize,
        num_layers: usize,
        num_heads: usize,
        drop_p: f32,
        safetensors_u8: &[u8],
    ) -> Result<Self, JsError> {
        Self::load_safetensors_and_build(safetensors_u8, |vocab_size| {
            LanguageModelBuilder::Transformer(TransformerLanguageModelOptions {
                n_embed,
                block_size,
                num_layers,
                num_heads,
                vocab_size,
                drop_p,
            })
        })
    }

    fn load_safetensors_and_build<F>(safetensors_u8: &[u8], factory: F) -> Result<Self, JsError>
    where
        F: FnOnce(usize) -> LanguageModelBuilder,
    {
        let device = Device::Cpu;
        let safetensors =
            candle_core::safetensors::BufferedSafetensors::new(safetensors_u8.into())?;
        let tokenizer_tensor = safetensors.load(TOKENIZER_VOCABULARY_KEY, &device)?;
        let tokenizer = Tokenizer::from_tensor(&tokenizer_tensor).map_err(e)?;
        let mut varmap = VarMap::new();
        let module = factory(tokenizer.len())
            .build_no_grad(&varmap, &device)
            .map_err(e)?;
        load_data_from_safetensors(&mut varmap, safetensors).map_err(e)?;
        Ok(Self {
            module: Arc::new(module),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

#[wasm_bindgen]
pub fn generate(
    safetensors_u8: &[u8],
    num_chars: usize,
    temperature: f32,
    seed: u64,
) -> Result<String, JsError> {
    let device = Device::Cpu;
    let safetensors = candle_core::safetensors::SliceSafetensors::new(safetensors_u8)?;
    let tokenizer_tensor = safetensors.load(TOKENIZER_VOCABULARY_KEY, &device)?;
    let tokenizer = Tokenizer::from_tensor(&tokenizer_tensor).map_err(e)?;
    let vocab_size = tokenizer.len();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // TODO: args should be passed in or pulled from safetensors.
    let block_size = 8;
    let model = TransformerLanguageModel::new(
        TransformerLanguageModelOptions {
            n_embed: 32,
            block_size,
            num_layers: 1,
            num_heads: 4,
            vocab_size,
            drop_p: 0.0,
        },
        vb,
    )
    .map_err(e)?;

    load_data_from_safetensors(&mut varmap, safetensors).map_err(e)?;

    let context = tokenizer.encode("\n").map_err(e)?;
    let mut language_generator =
        LanguageGenerator::new(&context, Box::new(model), block_size).map_err(e)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = String::with_capacity(num_chars);
    for _ in 0..num_chars {
        let char = language_generator
            .next_char(&mut rng, &tokenizer, temperature, &device)
            .map_err(e)?;
        result.push(char);
    }

    Ok(result)
}

// https://github.com/rustwasm/wasm-bindgen/issues/2970#issuecomment-2347845445
fn e(err: anyhow::Error) -> JsError {
    JsError::from(&*err)
}
