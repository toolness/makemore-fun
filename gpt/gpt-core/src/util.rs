use std::ops::Deref;

use anyhow::{Result, anyhow};
use candle_core::{
    Device, Tensor,
    backprop::GradStore,
    safetensors::{MmapedSafetensors, SliceSafetensors},
};
use candle_nn::VarMap;
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};

/// Uh, candle doesn't seem to have multinomial sampling built-in, so
/// we'll just implement something janky here.
///
/// We could consider using https://github.com/EricLBuehler/candle-sampling
/// instead.
pub fn multinomial(tensor: &Tensor, rng: &mut StdRng) -> Result<u32> {
    let vec: Vec<f32> = tensor.get(0)?.to_vec1()?;
    let mut choices: Vec<u32> = Vec::with_capacity(vec.len());
    let mut weights: Vec<f32> = Vec::with_capacity(vec.len());

    for (i, &prob) in vec.iter().enumerate() {
        if prob > 0.0 {
            choices.push(i as u32);
            weights.push(prob);
        }
    }

    let dist = WeightedIndex::new(&weights)?;

    Ok(choices[dist.sample(rng)])
}

/// Uh, candle doesn't have an easy way of comparing tensors for
/// equality so we'll do this.
pub fn assert_equal_tensors(a: Tensor, b: Tensor) -> Result<()> {
    // WHY IS THIS SO HARD????????????????
    let eq = a.eq(&b)?.flatten_all()?.to_vec1::<u8>()?;
    for item in eq {
        assert_eq!(item, 1);
    }
    Ok(())
}

pub fn count_params(varmap: &VarMap) -> usize {
    varmap
        .all_vars()
        .iter()
        .map(|var| var.as_tensor().elem_count())
        .sum()
}

pub fn print_gradient_info(varmap: &VarMap, gradients: &GradStore) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        let tensor = var.deref();
        if let Some(grad) = gradients.get(tensor) {
            let grad_squared = grad.sqr()?;
            let grad_norm: f32 = grad_squared.sum_all()?.sqrt()?.to_scalar()?;
            println!("gradient norm for {name}: {:.4}", grad_norm);
            if grad_norm > 10.0 {
                println!("  ⚠️  WARNING: Large gradient!");
            } else if grad_norm < 1e-6 {
                println!("  ⚠️  WARNING: Vanishing gradient!");
            }
        } else {
            println!("⚠️  WARNING: No gradient for {name}!");
        }
    }
    Ok(())
}

/// Ideally this would be a trait supported by all the
/// safetensors structs in candle.
pub trait SafetensorLoader {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor>;
}

impl SafetensorLoader for MmapedSafetensors {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dev)
    }
}

impl<'a> SafetensorLoader for SliceSafetensors<'a> {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dev)
    }
}

pub fn load_data_from_safetensors<T: SafetensorLoader>(
    varmap: &mut VarMap,
    safetensors: T,
) -> Result<()> {
    // This is mostly what VarMap::load() does, but that method is specific to
    // loading data from a file, while this isn't.
    let mut tensor_data = varmap.data().lock().unwrap();
    for (name, var) in tensor_data.iter_mut() {
        let data = safetensors.load_tensor(name, var.device())?;
        if let Err(err) = var.set(&data) {
            return Err(anyhow!("error setting {name} using safetensor data: {err}",));
        }
    }
    Ok(())
}
