globalThis.onmessage = function (e) {
  console.log(e.data);
  // Do some work
  globalThis.postMessage('Done with work');
}
