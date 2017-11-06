#![no_main]
#[macro_use] extern crate libfuzzer_sys;
#[macro_use] extern crate lazy_static;
extern crate yalp;
extern crate futures;
extern crate futures_cpupool;
extern crate tokio_timer;

use std::time::Duration;
use std::cell::UnsafeCell;

use futures::Future;
use futures_cpupool::CpuPool;
use tokio_timer::Timer;

use yalp::{LispExpr, State};
use yalp::parse::parse_lisp_string;

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

struct ExprWrapper(LispExpr);

unsafe impl Send for ExprWrapper {}
unsafe impl Sync for ExprWrapper {}

fn exec_command(s: &str) {
    let parse_result = parse_lisp_string(s, STATE.get_inner());

    match parse_result {
        Ok(expr) => {
            let pool = CpuPool::new_num_cpus();
            let timer = Timer::default();

            // a future that resolves to Err after a timeout
            let timeout = timer.sleep(Duration::from_millis(50))
                .then(|_| Err(()));

            let wrapped_expr = ExprWrapper(expr);
            let eval = pool.spawn_fn(move || {
                let e = wrapped_expr.0;
                yalp::evaluator::eval(e, STATE.get_inner()).map(|_| ()).map_err(|_| ())
            });

            // a future that resolves to one of the above values -- whichever
            // completes first!
            let _ = timeout.select(eval).wait();
        }
        Err(ref parse_err) => {
            println!("Parse error: {:?}", parse_err);
        }
    }
}

struct StateWrapper(UnsafeCell<State>);

impl StateWrapper {
    fn get_inner(&self) -> &mut State {
        unsafe {
            &mut *self.0.get()
        }
    }
}

unsafe impl Send for StateWrapper {}
unsafe impl Sync for StateWrapper {}

lazy_static! {
    static ref STATE: StateWrapper = {
        let mut state = State::default();

        for def in PRELUDE {
            let parse_res =
                parse_lisp_string(def, &mut state).expect("Prelude statement failed to parse!");
            yalp::evaluator::eval(parse_res, &mut state).expect("Prelude statement failed to execute!");
        }

        StateWrapper(UnsafeCell::new(state))
    };
}

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = ::std::str::from_utf8(data) {
        exec_command(s);
    }
});
