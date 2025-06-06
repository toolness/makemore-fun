import init, { generate } from "../pkg/web.js";

async function run() {
  // First up we need to actually load the Wasm file, so we use the
  // default export to inform it where the Wasm file is located on the
  // server, and then we wait on the returned promise to wait for the
  // Wasm to be loaded.
  //
  // It may look like this: `await init('./pkg/without_a_bundler_bg.wasm');`,
  // but there is also a handy default inside `init` function, which uses
  // `import.meta` to locate the Wasm file relatively to js file.
  //
  // Note that instead of a string you can also pass in any of the
  // following things:
  //
  // * `WebAssembly.Module`
  //
  // * `ArrayBuffer`
  //
  // * `Response`
  //
  // * `Promise` which returns any of the above, e.g. `fetch("./path/to/wasm")`
  //
  // This gives you complete control over how the module is loaded
  // and compiled.
  //
  // Also note that the promise, when resolved, yields the Wasm module's
  // exports which is the same as importing the `*_bg` module in other
  // modes
  await init();

  const stuff = await fetch("boop.safetensors");

  const uint8Array = new Uint8Array(await stuff.arrayBuffer());

  const text = generate(uint8Array, 500, 1.0, BigInt(Date.now()));
  console.log(text);
}

run();
