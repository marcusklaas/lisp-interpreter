extern crate yalp;
extern crate rustyline;

use yalp::{evaluate_lisp_expr, parse_lisp_string, State};

fn main() {
    let mut rl = rustyline::Editor::<()>::new();
    let mut state = State::new();

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
                            Ok(val) => println!("{}", val),
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
