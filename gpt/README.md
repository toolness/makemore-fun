Tiny shakespeare is taken from:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

To build the web version:

```
cargo install wasm-pack

# https://github.com/rust-random/getrandom/issues/208#issuecomment-2944376492
cargo install wasm-bindgen-cli

cargo install basic-http-server
cargo run --release -- --save=boop
cd web
npm run build
cd ..
basic-http-server
```

Then go to http://localhost:4000/web/.
