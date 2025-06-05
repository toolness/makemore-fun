use wasm_bindgen::prelude::*;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GptError {
    #[error("foooooo")]
    Foo,
}

#[wasm_bindgen]
pub fn add(a: u32, b: u32) -> Result<u32, JsError> {
    if a == 3 {
        return Err(GptError::Foo.into());
    }
    Ok(a + b)
}
