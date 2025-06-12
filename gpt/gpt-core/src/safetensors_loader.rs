use anyhow::{Result, anyhow};
use candle_core::{
    Device, Tensor,
    safetensors::{BufferedSafetensors, MmapedSafetensors, SliceSafetensors},
};
use candle_nn::VarMap;

/// Ideally this would be a trait supported by all the
/// safetensors structs in candle.
pub trait SafetensorsLoader {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor>;
}

impl SafetensorsLoader for MmapedSafetensors {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dev)
    }
}

impl<'a> SafetensorsLoader for SliceSafetensors<'a> {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dev)
    }
}

impl SafetensorsLoader for BufferedSafetensors {
    fn load_tensor(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dev)
    }
}

pub fn load_data_from_safetensors<T: SafetensorsLoader>(
    varmap: &mut VarMap,
    safetensors: &T,
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
