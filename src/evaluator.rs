use super::*;

enum Instr {
    EvalAndPush(LispExpr),
    PopAndWrap(usize),
    EvalFunction(Vec<LispExpr>),
    PopCondPush(LispExpr, LispExpr),
    PopAndSet(String),
    PopState,
    BindArguments(Vec<String>),
}

fn unitary_int<F: Fn(u64) -> LispValue>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> Result<(), EvaluationError> {
    match stack.pop().unwrap() {
        LispValue::Integer(i) => Ok(stack.push(f(i))),
        _ => return Err(EvaluationError::ArgumentTypeMismatch),
    }
}

pub fn eval<'e>(expr: &'e LispExpr, init_state: &mut State) -> Result<LispValue, EvaluationError> {
    let mut instructions = Vec::new();
    let mut return_values = Vec::new();
    let mut states: Vec<State> = Vec::new();
    let mut state = init_state.clone();

    instructions.push(Instr::EvalAndPush(expr.clone()));

    while let Some(instr) = instructions.pop() {
        match instr {
            Instr::PopState => {
                state = states.pop().unwrap();
            }
            Instr::EvalAndPush(expr) => {
                match expr {
                    LispExpr::Integer(i) => {
                        return_values.push(LispValue::Integer(i));
                    }
                    LispExpr::OpVar(ref n) => {
                        let val = state.get_variable_value(n);
                        return_values.push(val);
                    }
                    // This is actually a function call - we should
                    // probably rename it.
                    LispExpr::SubExpr(mut expr_vec) => {
                        // step 1: remove head expression
                        let head_expr = expr_vec.remove(0);

                        // step 2: queue function evaluation with tail
                        instructions.push(Instr::EvalFunction(expr_vec));

                        // step 3: queue evaluation of head
                        instructions.push(Instr::EvalAndPush(head_expr));
                    }
                }
            }
            Instr::PopAndWrap(count) => {
                let len = return_values.len();
                let new_vec = return_values.split_off(len - count);
                return_values.push(LispValue::SubValue(new_vec));
            }
            // Pops a function off the value stack and applies it
            Instr::EvalFunction(mut expr_list) => {
                let head = return_values.pop().unwrap();
                match head {
                    LispValue::Function(f) => {
                        match f {
                            LispFunc::BuiltIn(func_name) => {
                                match (&func_name[..], expr_list.len()) {
                                    ("list", _) => {
                                        instructions.push(Instr::PopAndWrap(expr_list.len()));
                                        for expr in expr_list.into_iter().rev() {
                                            instructions.push(Instr::EvalAndPush(expr));
                                        }
                                    }
                                    // ("zero?", 1) => {
                                    //     instructions.push(Instr::CheckZero);
                                    //     instructions.push(Instr::EvalAndPush(expr_list.remove(0)));
                                    // }
                                    // ("add1", 1) => {
                                    //     instructions.push(Instr::AddOne);
                                    //     instructions.push(Instr::EvalAndPush(expr_list.remove(0)));
                                    // }
                                    // ("null?", 1) => {
                                    //     instructions.push(Instr::CheckNull);
                                    //     instructions.push(Instr::EvalAndPush(expr_list.remove(0)));
                                    // }
                                    ("cond", 3) => {
                                        let (truth_value, true_expr, false_expr) = (expr_list.remove(0), expr_list.remove(0), expr_list.remove(0));
                                        // Queue condition evaluation
                                        instructions.push(Instr::PopCondPush(true_expr, false_expr));
                                        // Queue truth value
                                        instructions.push(Instr::EvalAndPush(truth_value));
                                    }
                                    ("lambda", 2) => {
                                        match expr_list.remove(0) {
                                            LispExpr::SubExpr(arg_vec) => {
                                                let f = LispFunc::Custom {
                                                    state: state.clone(),
                                                    args: arg_vec.into_iter().map(|expr| match expr {
                                                        LispExpr::OpVar(name) => Ok(name),
                                                        _ => Err(EvaluationError::MalformedDefinition),
                                                    }).collect::<Result<Vec<_>, _>>()?,
                                                    body: expr_list.remove(0),
                                                };

                                                return_values.push(LispValue::Function(f));
                                            }
                                            _ => return Err(EvaluationError::ArgumentTypeMismatch),
                                        }
                                    }
                                    ("define", 2) => {
                                        match expr_list.remove(0) {
                                            LispExpr::OpVar(name) => {
                                                instructions.push(Instr::PopAndSet(name));
                                                instructions.push(Instr::EvalAndPush(expr_list.remove(0)));
                                            }
                                            _ => return Err(EvaluationError::ArgumentTypeMismatch),
                                        }
                                    }
                                    _ => return Err(EvaluationError::UnknownVariable(func_name)),
                                }
                            }
                            LispFunc::Custom {
                                state: mut closure,
                                args,
                                body,
                            } => {
                                ::std::mem::swap(&mut closure, &mut state);
                                states.push(closure);
                                instructions.push(Instr::PopState);
                                instructions.push(Instr::EvalAndPush(body));
                                instructions.push(Instr::BindArguments(args));
                                for expr in expr_list.into_iter().rev() {
                                    instructions.push(Instr::EvalAndPush(expr));
                                }
                            }
                        }
                    }
                    _ => return Err(EvaluationError::NonFunctionApplication),
                }
            }
            Instr::BindArguments(name_mapping) => {
                for arg_name in &name_mapping {
                    state.set_variable(arg_name, return_values.pop().unwrap());
                }
            }
            Instr::PopCondPush(true_expr, false_expr) => {
                if let LispValue::Truth(b) = return_values.pop().unwrap() {
                    let next_instr = if b { true_expr } else { false_expr };
                    instructions.push(Instr::EvalAndPush(next_instr));
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::PopAndSet(var_name) => {
                let val = return_values.pop().unwrap();
                state.set_variable(&var_name, val.clone());
                return_values.push(val);
            }
        }
    }

    *init_state = state;
    assert!(return_values.len() == 1);
    Ok(return_values.pop().unwrap())
}
