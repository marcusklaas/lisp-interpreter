#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]

extern crate yalp;
extern crate rustyline;

use yalp::evaluator::State;
use yalp::parse::parse_lisp_string;

const PRELUDE: &'static [&'static str] = &[
    "(define closure (lambda (x) (lambda (y) (add x y))))",
    "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
    "(define mult (lambda (x y) (cond (zero? y) 0 (add (mult x (sub1 y)) x))))",
    "(define filter (lambda (f xs) (cond (null? xs) (list) (cond (f (car xs)) (cons (car xs) (filter f (cdr xs))) (filter f (cdr xs))))))",
    "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
    "(define not (lambda (t) (cond t #f #t)))",
    "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
    "(define and (lambda (t1 t2) (cond t1 t2 #f)))",
    "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
    "(define range (lambda (start end) (cond (> end start) (cons end (range start (sub1 end))) (list start))))",
    "(define sort (lambda (l) (cond (null? l) l (append (cons (car l) (sort (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort (filter (lambda (x) (> x (car l))) l))))))",
];

fn main() {
    let mut rl = rustyline::Editor::<()>::new();
    let mut state = State::new();

    if ::std::env::args().skip(1).collect::<Vec<String>>() != vec!["--no-prelude"] {
        for def in PRELUDE {
            let parse_res = parse_lisp_string(def).expect("Prelude statement failed to parse!");
            yalp::evaluator::eval(&parse_res, &mut state)
                .expect("Prelude statement failed to execute!");
        }
    }

    loop {
        match rl.readline("> ") {
            Ok(ref line) if line == ":quit" => break,
            Ok(ref line) if line == ":state" => {
                println!("{:?}", &state);
                rl.add_history_entry(line);
            }
            Ok(ref line) => {
                rl.add_history_entry(line);

                let parse_result = parse_lisp_string(line);

                match parse_result {
                    Ok(ref expr) => {
                        //match evaluate_lisp_expr(expr, &mut state) {
                        match yalp::evaluator::eval(expr, &mut state) {
                            Ok(val) => {
                                println!("{}", &val);
                                state.set_variable(":last", val);
                            }
                            Err(eval_err) => println!("Evaluation error: {:?}", eval_err),
                        }
                    }
                    Err(ref parse_err) => {
                        println!("Parse error: {:?}", parse_err);
                    }
                }
            }
            Err(..) => {
                break;
            }
        }
    }
}
