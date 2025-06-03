use anyhow::Result;
use candle_core::Tensor;
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
    let mut weights: Vec<u32> = Vec::with_capacity(vec.len());

    for (i, prob) in vec.iter().enumerate() {
        let weight = (prob * 100.0) as u32;
        if weight > 0 {
            choices.push(i as u32);
            weights.push(weight);
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
