use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use gpt_core::{
    language_model::LanguageGenerator, tokenizer::Tokenizer,
    transformer_language_model::TransformerLanguageModel,
};
use rand::{SeedableRng, rngs::StdRng};
use wasm_bindgen::prelude::*;

use anyhow::anyhow;

#[wasm_bindgen]
pub fn generate(
    safetensors_u8: &[u8],
    num_chars: usize,
    temperature: f32,
    seed: u64,
) -> Result<String, JsError> {
    let device = Device::Cpu;
    let safetensors = candle_core::safetensors::SliceSafetensors::new(safetensors_u8)?;

    // TODO: `BUFFER.tokenizer_vocabulary` should be a const in gpt-core
    let tokenizer_tensor = safetensors.load("BUFFER.tokenizer_vocabulary", &device)?;

    let tokenizer = Tokenizer::from_tensor(&tokenizer_tensor).map_err(e)?;
    let vocab_size = tokenizer.len();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // TODO: args should be passed in or pulled from safetensors.
    let block_size = 8;
    let model =
        TransformerLanguageModel::new(32, block_size, 1, 4, vocab_size, 0.0, vb).map_err(e)?;

    {
        let mut tensor_data = varmap.data().lock().unwrap();
        for (name, var) in tensor_data.iter_mut() {
            let data = safetensors.load(name, var.device())?;
            if let Err(err) = var.set(&data) {
                return Err(e(anyhow!(
                    "error setting {name} using safetensor data: {err}",
                )));
            }
        }
    }

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
