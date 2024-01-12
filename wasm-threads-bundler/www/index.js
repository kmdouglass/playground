let worker = new Worker(new URL('./worker.js', import.meta.url));

worker.postMessage('Hello World');

worker.onmessage = function (e) {
    console.log('Message received from worker:', e.data);
    worker.terminate();
}
