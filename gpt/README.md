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

## Web deployment

This will build and deploy the web version to GitHub pages under
the `/makemore-fun/` subdirectory.

```
npm run deploy
```

## Credits

Tiny shakespeare is taken from:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
