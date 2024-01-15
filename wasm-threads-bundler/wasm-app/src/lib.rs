use wasm_bindgen::prelude::*;

/// The application state
struct AppState {
    counter: u32,
}

impl AppState {
    fn new() -> Self {
        Self { counter: 0 }
    }

    fn counter(&mut self) -> u32 {
        let ret = self.counter;
        self.counter += 1;
        ret
    }
}

/// Wraps the Rust application state and exposes it to JavaScript
#[wasm_bindgen]
pub struct WasmApp {
    state: AppState,
}

#[wasm_bindgen]
#[allow(non_snake_case)]
impl WasmApp {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: AppState::new(),
        }
    }

    pub fn counter(&mut self) -> JsValue {
        JsValue::from(self.state.counter())
    }
}
