extern crate yalp;
extern crate rustyline;

use yalp::eval::{evaluate_lisp_expr, State};
use yalp::parse::parse_lisp_string;

const PRELUDE: &'static [&'static str] = &[
    "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
    "(define mult (lambda (x y) (cond (zero? y) 0 (add x (mult x (sub1 y))))))",
    "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
];

fn main() {
    let mut rl = rustyline::Editor::<()>::new();
    let mut state = State::new();

    for def in PRELUDE {
        let parse_res = parse_lisp_string(def).expect("Prelude statement failed to parse!");
        evaluate_lisp_expr(&parse_res, &mut state).expect("Prelude statement failed to execute!");
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
                        match evaluate_lisp_expr(expr, &mut state) {
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
