# lisp-interpreter 
An bare bones lisp parser &amp; interpreter.

[![Build Status](https://travis-ci.org/marcusklaas/lisp-interpreter.svg?branch=master2)](https://travis-ci.org/marcusklaas/lisp-interpreter)

## Features
Parses and evaluates simple lisp-like statements. Its features include lambdas, closures and currying.
All data is immutable and the only types availables are unsigned integers, booleans, functions and lists.
The interpreter has simulates its own stack, so recursion is not bounded by the stack size of the interpreter.

Available built-in functions:

| function name  | arguments | output type |
|---|---|---|
| add1 | int | int |
| sub1 | positive int | int |
| zero? | int | bool |
| car | non-empty list | * |
| cdr | non-empty list | list |
| null? | list | bool |
| cons | *, list | list |
| define | name, * | empty list |
| lambda | list, * | function |
| cond | bool, *, * | * |
| list | * | list |

Further, the main binary introduces some convenience functions: `add`, `mult`, `map`, `filter`, `>`, `sort`, `append`, `not` and `and`.
These are defined in terms of the built-in functions above.

Example evaluations:
```
> (add1 (add1 3))
5
> (define sub (lambda (x y) (cond (zero? y) x (sub (sub1 x) (sub1 y)))))
()
> (list sub)
(func[2 -> t(func[cond] (func[zero?] $1) $0 t(sub (func[sub1] $0) (func[sub1] $1)))])
> (sub 5 3)
2
> (sub 10)
func[1 -> t(func[2 -> t(func[cond] (func[zero?] $1) $0 t(sub (func[sub1] $0) (func[sub1] $1)))] 1 $0)]
> (:last 1)
9
> (sub 1 5)
Evaluation error: SubZero
```

## Installation
Make sure you have rust installed:
```bash
$ curl https://sh.rustup.rs -sSf | sh
```
Clone the repo
```bash
$ git clone https://github.com/marcusklaas/lisp-interpreter.git
```
Build using cargo
```bash
$ cd lisp-interpreter
$ cargo build --release
```
To start the command-line REPL, run
```bash
$ cargo run --release
```
Execute the test suite using
```bash
$ cargo test
```
