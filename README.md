# lisp-interpreter 
A bare bones lisp parser &amp; interpreter.

[![Build Status](https://travis-ci.org/marcusklaas/lisp-interpreter.svg?branch=master2)](https://travis-ci.org/marcusklaas/lisp-interpreter)

## Features
Parses and evaluates simple lisp-like statements. Its features include lambdas, closures and currying.
All data is immutable and the only types availables are unsigned integers, booleans, functions and lists.
The interpreter simulates its own stack, so recursion is not bounded by the stack size of the interpreter.

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
| int? | * | bool |
| bool? | * | bool |
| list? | * | bool |
| fun? | * | bool |

Further, the main binary introduces some convenience functions, including `add`, `mult`, `map`, `filter`, `>`, `sort`, `append`, `not` and `and`.
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

## Some technical details
This lisp interpreter has a fairly simple design. It goes through the usual steps in the interpretation: it parses input strings into a sequence of tokens, builds an AST, does some (light) analysis on this AST to produce a "finalized" AST and finally compiles this into bytecode. The unit of compilation and execution is the function. Execution is done by keeping a stack of values, onto which function arguments and return value are pushed and popped. Every time a function is called, a new stack reference is created, which contains a pointer to the function's bytecode, an instruction pointer, and the position of its arguments on the stack. When a function calls another function, its own stack reference is pushed onto the reference stack (unless analysis showed that this call is a tail-call) and the current stack reference is replaced by that of the callee. Whenever a function returns, the stack reference of the calling function is popped off the reference stack and execution continues there.

This interpreter does not use a garbage colllector to keep the design simple. Functions are reference counted and all other values are cloned or moved. Mutation of values is not possible, although mutation does happen at execution time as an optimization. There is a single environment that holds definitions. Definitions cannot be overwritten.

Because the set of buitl-in functions is so sparse, writing performant code for this interpreter is generally not possible. However, it does perform elementary operations relatively quickly. For example, the prelude function `add`, which recursively adds 1 to the first argument and subtracts 1 from the second until the second argument is zero is about twice as fast as the following loop in PHP 7.1.8:
```php
function add($a, $b) {
    while($b > 0) {
        $a += 1;
        $b -= 1;
    }
    return $a;
}
```
However, it is about five times slower than a similar loop in V8.

Code that heavily relies on list operations should be reasonably fast, as it is internally represented by Vectors and not by Linked Lists, as is common for many other interpreters. Again, since there are few built-in functions, common operations like appending lists is not optimized (it does a list reversal as an intermediate step!) and is way slower than it could be with a better optimizing compiler. 

Adding loop analysis and mathy substitution rules (repeated increments = addition, repeated addition = multiplication, etc.) would be a cool project that could significantly speed up common operations.
