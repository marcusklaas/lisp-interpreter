extern crate yalp;
extern crate rustyline;

use yalp::{evaluate_lisp_expr, parse_lisp_string, State};

fn main() {
    let mut rl = rustyline::Editor::<()>::new();
    let mut state = State::new();

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                if line == "quit" {
                    break;
                }
                rl.add_history_entry(&line);

                let parse_result = parse_lisp_string(&line);
                println!("Parse result: {:?}", parse_result);

                if let Ok(ref expr) = parse_result {
                    let eval = evaluate_lisp_expr(expr, &mut state);
                    println!("Evaluation result: {:?}", eval);
                }
            }
            Err(..) => {
                break;
            }
        }
    }
}
