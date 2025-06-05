Tiny shakespeare is taken from:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

To build the web version:

```
cargo install wasm-pack
cargo install basic-http-server
wasm-pack build web --target web
cd web
basic-http-server
```

Then go to http://localhost:4000.
