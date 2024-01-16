# WebAssembly Threads and Bundlers

A minimal example of how to deploy a Javascript + Rust/Wasm app that uses WebAssembly threads via a bundler.

## Summary

The purpose of this example is to demonstrate how to build a Rust/Wasm app with WebAssembly support that is deployed via a bundler. The demonstration is a minimal example.

I spent a few weeks researching how to do this, which, in hindsight, was longer than it ought to have been. This is because the Rust/Wasm docs are slightly misleading in the explanation about deploying Wasm apps via a bundler. Specifically, [the wasm_bindgen guide](https://rustwasm.github.io/wasm-bindgen/examples/raytrace.html) states that

> Setting up a threaded environment is a bit wonky and doesn't feel smooth today. For example --target bundler is unsupported and very specific shims are required on both the main thread and worker threads. These are possible to work with but are somewhat brittle since there's no standard way to spin up web workers as wasm threads.

Additionally, all examples that I found that built Wasm apps with `--target web` were not deployed with a bundler, leaving me to wonder whether a bundler was incompatible thiis target. Happily, the demo project of [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon) showed that it was indeed possible to use WebAssembly threads in JS + Rust/Wasm apps deployed with a bundler.

## Explanation

See [www/package.json](www/package.json) for the build scripts. The `npm run build:wasm` command builds the Wasm artifacts and puts them inside the `www/pkg` directory where they can be bundled by Webpack.

Because we built the artifacts with `--target web` we need to initialize the Wasm module ourselves in both the main module and the worker:

### Main module

```javascript
import init, { WasmApp } from './pkg/wasm_app.js';

init().then(() => {
    // Application code goes here
}
```

### Web worker

```javascript
import init, { isEven } from './pkg/wasm_app.js';

async function init_wasm_in_worker() {
  console.log('Initializing worker');

  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await init();
  
  // ...
}
init_wasm_in_worker();
```

The `init` function is the default export of the `wasm_app.js` file that is created by webpack. At the time of this writing, this export is just a JS shim to instantiate a WebAssembly module, `__wbg_init`.

## Installation

### Install nightly Rust

WebAssembly threads are only supported on nightly at the time of this writing (2024/01/16).

```console
rustup component add rust-src --toolchain nightly-2024-01-12-x86_64-unknown-linux-gnu
```

### Install Node modules

From inside the `www` folder:

```console
npm install
```

## Build the app

### Bundle everything into `www/dist`

From inside the `www` folder:

```console
npm run build
```

### Build just the Wasm artifacts

From inside the `www` folder:

```console
npm run build:wasm
```

## Develop

From inside the `www` folder:

```console
npm run start
```
