#![feature(link_args)]
#![allow(unused_attributes)]

extern crate yalp;

use yalp::State;
use yalp::parse::parse_lisp_string;

use std::mem::{forget, transmute};
use std::os::raw::c_char;
use std::ffi::{CStr, CString};

#[link_args = "-s EXPORTED_FUNCTIONS=['_exec_command','_create_state','_free_state']"]
extern "C" {}

const PRELUDE: &'static [&'static str] = &[
    "(define closure (lambda (x) (lambda (y) (add x y))))",
    "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
    "(define mult (lambda (x y) (cond (zero? y) 0 (add (mult x (sub1 y)) x))))",
    "(define filter (lambda (f xs) (cond (null? xs) xs (cond (f (car xs)) (cons (car xs) (filter f (cdr xs))) (filter f (cdr xs))))))",
    "(define map (lambda (f xs) (cond (null? xs) xs (cons (f (car xs)) (map f (cdr xs))))))",
    "(define not (lambda (t) (cond t #f #t)))",
    "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
    "(define and (lambda (t1 t2) (cond t1 t2 #f)))",
    "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
    "(define range (lambda (start end) (cond (> end start) (cons end (range start (sub1 end))) (list start))))",
    "(define sort (lambda (l) (cond (null? l) l (append (cons (car l) (sort (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort (filter (lambda (x) (> x (car l))) l))))))",
    "(define or (lambda (x y) (cond x #t y)))",
    "(define zip (lambda (x y) (cond (or (null? x) (null? y)) (list) (cons (list (car x) (car y)) (zip (cdr x) (cdr y))))))",
    "(define map2 (lambda (f l) (cond (null? l) l (cons (f (car (cdr (car l))) (car (car l))) (map2 f (cdr l))))))",
    "(define reverse (lambda (l) (cond (null? l) l (append (list (car l)) (reverse (cdr l))))))",
    "(define !! (lambda (l i) (cond (zero? i) (car l) (!! (cdr l) (sub1 i)))))",
    "(define foldr (lambda (f xs init) (cond (null? xs) init (foldr f (cdr xs) (f init (car xs))))))",
];

fn exec(s: &str, state: &mut State) -> String {
    let parse_result = parse_lisp_string(s, state);
    let last_intern = state.intern(":last");

    match parse_result {
        Ok(expr) => match yalp::evaluator::eval(expr, state) {
            Ok(val) => {
                let res = val.pretty_print(state, 0);
                state.set_variable(last_intern, val, true).unwrap();
                res
            }
            Err(eval_err) => format!("Evaluation error: {:?}", eval_err),
        },
        Err(ref parse_err) => format!("Parse error: {:?}", parse_err),
    }
}

#[no_mangle]
pub extern "C" fn create_state() -> *mut State {
    let mut state = State::default();

    for def in PRELUDE {
        let parse_res =
            parse_lisp_string(def, &mut state).expect("Prelude statement failed to parse!");
        yalp::evaluator::eval(parse_res, &mut state).expect("Prelude statement failed to execute!");
    }

    unsafe { transmute(Box::new(state)) }
}

#[no_mangle]
pub extern "C" fn free_state(state: *mut State) {
    unsafe { transmute::<*mut _, Box<State>>(state) };
}

#[no_mangle]
pub extern "C" fn exec_command(command: *const c_char, state: *mut State) -> *mut c_char {
    let rust_string = unsafe { CStr::from_ptr(command).to_str().unwrap() };
    let mut state: Box<State> = unsafe { transmute(state) };

    let res = exec(rust_string, &mut *state);
    forget(state);
    CString::new(res).unwrap().into_raw()
}

fn main() {
    // intentionally blank
}
