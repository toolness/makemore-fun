Tiny shakespeare is taken from:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

To build the web version:

```
cargo install wasm-pack

# https://github.com/rust-random/getrandom/issues/208#issuecomment-2944376492
cargo install wasm-bindgen-cli

cargo run --release -- --save=web/boop
cd web
npm install
npm run wasm
npm run dev
```

Then go to http://localhost:5173/.
