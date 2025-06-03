use anyhow::{Result, anyhow};
use std::fmt::Display;

use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda => write!(f, "CUDA"),
            Device::Metal => write!(f, "Metal"),
        }
    }
}

impl Device {
    pub fn to_candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda => {
                if cfg!(feature = "cuda") {
                    Ok(candle_core::Device::new_cuda(0)?)
                } else {
                    return Err(anyhow!(
                        "CUDA is not supported in this build, you need to compile with the 'cuda' feature!"
                    ));
                }
            }
            Device::Metal => {
                if cfg!(feature = "metal") {
                    Ok(candle_core::Device::new_metal(0)?)
                } else {
                    return Err(anyhow!(
                        "Metal is not supported in this build, you need to compile with the 'metal' feature!"
                    ));
                }
            }
        }
    }
}
