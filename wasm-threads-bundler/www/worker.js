import init, { isEven } from './pkg/wasm_app.js';

async function init_wasm_in_worker() {
  console.log('Initializing worker');

  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await init();
  console.log('Worker intialized');

  // Set callback to handle messages passed to the worker.
  self.onmessage = async e => {
      console.log('Message received from main thread: ', e.data);
      let result = isEven(e.data);

      // Send response back to be handled by callback in main thread.
      self.postMessage(result);
  };
};

init_wasm_in_worker();
