#!/bin/bash

cargo build --target=wasm32-unknown-emscripten --release
find target/wasm32-unknown-emscripten/release/deps -type f -name "*.wasm" | xargs -I {} cp {} site/site.wasm
find target/wasm32-unknown-emscripten/release/deps -type f ! -name "*.asm.js" -name "*.js" | xargs -I {} cp {} site/site.js
