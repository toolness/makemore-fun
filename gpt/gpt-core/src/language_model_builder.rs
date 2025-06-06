use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};

use crate::{
    bigram_language_model::BigramLanguageModel,
    transformer_language_model::{TransformerLanguageModel, TransformerLanguageModelOptions},
};

#[derive(Copy, Clone)]
pub enum LanguageModelBuilder {
    Bigram(usize),
    Transformer(TransformerLanguageModelOptions),
}

impl LanguageModelBuilder {
    pub fn build(self, vb: VarBuilder) -> Result<Box<dyn Module>> {
        match self {
            Self::Bigram(vocab_size) => Ok(Box::new(BigramLanguageModel::new(vocab_size, vb)?)),
            Self::Transformer(options) => Ok(Box::new(TransformerLanguageModel::new(options, vb)?)),
        }
    }

    pub fn build_no_grad(
        self,
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
        self.build(VarBuilder::from_tensors(detached_vars, DType::F32, device))
    }
}
