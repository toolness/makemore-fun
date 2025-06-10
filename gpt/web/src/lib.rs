use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use gpt_core::{
    char_tokenizer::{CHAR_TOKENIZER_VOCABULARY_KEY, CharTokenizer},
    language_model::LanguageGenerator,
    language_model_builder::LanguageModelBuilder,
    transformer_language_model::TransformerLanguageModelOptions,
    util::load_data_from_safetensors,
};
use rand::{SeedableRng, rngs::StdRng};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmLanguageModel {
    varmap: VarMap,
    builder: LanguageModelBuilder,
    tokenizer: CharTokenizer,
}

#[wasm_bindgen]
impl WasmLanguageModel {
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
        let safetensors = candle_core::safetensors::SliceSafetensors::new(safetensors_u8.into())?;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let tokenizer_tensor = safetensors.load(CHAR_TOKENIZER_VOCABULARY_KEY, &device)?;
        let tokenizer = CharTokenizer::from_tensor(&tokenizer_tensor).map_err(e)?;
        let builder = factory(tokenizer.len());

        builder.build(vb).map_err(e)?;
        load_data_from_safetensors(&mut varmap, &safetensors).map_err(e)?;

        Ok(Self {
            varmap,
            builder,
            tokenizer,
        })
    }

    #[wasm_bindgen]
    pub fn create_generator(
        &self,
        seed: u64,
        initial_context: &str,
    ) -> Result<WasmLanguageGenerator, JsError> {
        let device = Device::Cpu;
        let model = self
            .builder
            .clone()
            .build_no_grad(&self.varmap, &device)
            .map_err(e)?;

        let context = self.tokenizer.encode_safe(initial_context);
        let block_size = model.block_size();
        let generator = LanguageGenerator::new(&context, model, block_size).map_err(e)?;
        Ok(WasmLanguageGenerator::create(
            seed,
            generator,
            self.tokenizer.clone(),
        ))
    }
}

#[wasm_bindgen]
pub struct WasmLanguageGenerator {
    rng: StdRng,
    generator: LanguageGenerator,
    tokenizer: CharTokenizer,
    device: Device,
}

#[wasm_bindgen]
impl WasmLanguageGenerator {
    fn create(seed: u64, generator: LanguageGenerator, tokenizer: CharTokenizer) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            generator,
            tokenizer,
            device: Device::Cpu,
        }
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self, temperature: f32) -> Result<char, JsError> {
        self.generator
            .next_char(&mut self.rng, &self.tokenizer, temperature, &self.device)
            .map_err(e)
    }
}

// https://github.com/rustwasm/wasm-bindgen/issues/2970#issuecomment-2347845445
fn e(err: anyhow::Error) -> JsError {
    JsError::from(&*err)
}
