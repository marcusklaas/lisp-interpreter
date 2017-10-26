WASM lisp interpreter
=====================

This is a REPL for a lisp interpreter written in rust for the browser. A demo can be found at [marcusklaas.nl/lisp](https://marcusklaas.nl/lisp/). A polyfill for browser which do not support webAssembly yet should be included.

How to build
------------

By far the easiest way to compile to webassembly is to use [Wargo](https://github.com/lord/wargo/):

 - Install [rust](https://rust-lang.org)
 - Install wargo: `$ sudo npm install -g wargo`
 - Run the build script `$ ./build.sh`
 - Open `site/index.html` in a browser
