import init, { WasmApp } from './pkg/wasm_app.js';

init().then(() => {
    let worker = new Worker(new URL('./worker.js', import.meta.url));
    worker.onmessage = function (e) {
        console.log('Message received from worker: ', e.data);
    }
    
    let app = new WasmApp();
    let counter = app.counter();

    console.log('Sending message to worker: ', counter);
    // sleep for 0.5 second to allow worker to initialize
    setTimeout(() => worker.postMessage(counter), 500);
    setTimeout(() => worker.postMessage(app.counter()), 500);
});
