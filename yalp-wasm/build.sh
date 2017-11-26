#!/bin/bash

cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/yalp-wasm.wasm site/site.wasm
