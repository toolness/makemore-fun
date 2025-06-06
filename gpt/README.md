## Web quick start

For the web version, you will need:

* Git LFS
* Node and NPM

To build the web version:

```
# This is only really needed if you installed Git LFS *after* cloning the repo.
git lfs fetch
git lfs checkout

cargo install wasm-pack

# https://github.com/rust-random/getrandom/issues/208#issuecomment-2944376492
cargo install wasm-bindgen-cli

cd web
npm install
npm run wasm
npm run dev
```

Then go to http://localhost:5173/.

## Credits

Tiny shakespeare is taken from:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
