use super::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct State {
    pub bound: HashMap<String, LispValue>,
}

impl State {
    pub fn new() -> State {
        State {
            bound: vec![
                "zero?",
                "add1",
                "sub1",
                "cons",
                "null?",
                "cdr",
                "car",
                "cond",
                "define",
                "lambda",
                "list",
            ].into_iter()
                .map(|x| (x, LispValue::Function(LispFunc::BuiltIn(x.into()))))
                .into_iter()
                .chain(
                    vec![
                        ("#t", LispValue::Boolean(true)),
                        ("#f", LispValue::Boolean(false)),
                    ].into_iter(),
                )
                .map(|(var_name, val)| (var_name.into(), val))
                .collect(),
        }
    }

    pub fn get_variable_value(&self, var_name: &str) -> Option<LispValue> {
        self.bound.get(var_name).cloned()
    }

    pub fn set_variable(&mut self, var_name: &str, val: LispValue) {
        self.bound.insert(var_name.into(), val);
    }
}

enum Instr {
    EvalAndPush(LispExpr),
    // Function and argument vector, Tail call
    EvalFunction(Vec<LispExpr>, bool),
    PopCondPush(LispExpr, LispExpr),
    PopAndSet(String),
    PopState,
    // Function, Argument Count, Tail call
    EvalFunctionEager(LispFunc, usize, bool),
    SetStackPointer(usize),
}

fn unitary_int<F: Fn(u64) -> Result<LispValue, EvaluationError>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> Result<(), EvaluationError> {
    match stack.pop().unwrap() {
        LispValue::Integer(i) => Ok(stack.push(f(i)?)),
        _ => {
            return Err(EvaluationError::ArgumentTypeMismatch);
        }
    }
}

fn unitary_list<F: Fn(Vec<LispValue>) -> Result<LispValue, EvaluationError>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> Result<(), EvaluationError> {
    match stack.pop().unwrap() {
        LispValue::SubValue(v) => Ok(stack.push(f(v)?)),
        _ => return Err(EvaluationError::ArgumentTypeMismatch),
    }
}

macro_rules! do_nothing {
    ( $y:ident, $x:expr ) => {{$x}};
}

macro_rules! destructure {
    ( $y:ident, [ $( $i:ident ),* ], $body:expr ) => {
        {
            if let ($( Some($i), )* None) = {
                let mut iter = $y.into_iter();
                ( $( do_nothing!($i, iter.next()), )* iter.next() )
            } {
                Ok($body)
            } else {
                Err(EvaluationError::ArgumentCountMismatch)
            }
        }
    };
}

pub fn eval<'e>(expr: &'e LispExpr, init_state: &mut State) -> Result<LispValue, EvaluationError> {
    let mut return_values: Vec<LispValue> = Vec::new();
    let mut state = init_state.clone();
    let mut instructions = vec![Instr::EvalAndPush(expr.clone())];
    let mut stack_pointers = vec![0usize];
    // let mut buffer: Vec<LispValue> = Vec::new();

    while let Some(instr) = instructions.pop() {
        match instr {
            Instr::SetStackPointer(p) => {
                stack_pointers.push(p);
            }
            Instr::PopState => {
                let val = return_values.pop().unwrap();
                let pointer = stack_pointers.pop().unwrap();
                return_values.truncate(pointer);
                return_values.push(val);
            }
            Instr::EvalAndPush(expr) => {
                match expr {
                    LispExpr::Argument(offset) => {
                        let pointer = stack_pointers.last().unwrap();
                        let index = pointer + offset;
                        let value: LispValue = (&return_values[index]).clone();
                        return_values.push(value);
                    }
                    LispExpr::Value(v) => {
                        return_values.push(v);
                    }
                    LispExpr::OpVar(ref n) => {
                        if let Some(v) = state.get_variable_value(n) {
                            return_values.push(v);
                        } else {
                            return Err(EvaluationError::UnknownVariable(n.clone()));
                        }
                    }
                    LispExpr::Call(mut expr_vec, is_tail_call) => {
                        // step 1: remove head expression
                        let head_expr = expr_vec.remove(0);

                        // step 2: queue function evaluation with tail
                        instructions.push(Instr::EvalFunction(expr_vec, is_tail_call));

                        // step 3: queue evaluation of head
                        instructions.push(Instr::EvalAndPush(head_expr));
                    }
                }
            }
            // Pops a function off the value stack and applies it
            Instr::EvalFunction(expr_list, is_tail_call) => {
                match return_values.pop().unwrap() {
                    LispValue::Function(LispFunc::BuiltIn(ref name)) if name == "cond" => {
                        destructure!(expr_list, [boolean, true_expr, false_expr], {
                            // Queue condition evaluation
                            instructions.push(Instr::PopCondPush(true_expr, false_expr));
                            // Queue Boolean value
                            instructions.push(Instr::EvalAndPush(boolean));
                        })?
                    }
                    LispValue::Function(LispFunc::BuiltIn(ref name)) if name == "lambda" => {
                        destructure!(
                            expr_list,
                            [arg_list, body],
                            match arg_list {
                                LispExpr::Call(arg_vec, _is_tail_call) => {
                                    let args = arg_vec
                                        .into_iter()
                                        .map(|expr| match expr {
                                            LispExpr::OpVar(name) => Ok(name),
                                            _ => Err(EvaluationError::MalformedDefinition),
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;

                                    let stack_pointer = *stack_pointers.last().unwrap();
                                    let walked_body =
                                        body.replace_args(&return_values[stack_pointer..]);

                                    let f = LispFunc::new_custom(args, walked_body, &state);

                                    return_values.push(LispValue::Function(f));
                                }
                                _ => {
                                    return Err(EvaluationError::ArgumentTypeMismatch);
                                }
                            }
                        )?
                    }
                    LispValue::Function(LispFunc::BuiltIn(ref name)) if name == "define" => {
                        destructure!(
                            expr_list,
                            [var_name, definition],
                            match var_name {
                                LispExpr::OpVar(name) => {
                                    instructions.push(Instr::PopAndSet(name));
                                    instructions.push(Instr::EvalAndPush(definition));
                                }
                                _ => {
                                    return Err(EvaluationError::ArgumentTypeMismatch);
                                }
                            }
                        )?
                    }
                    // Eager argument evaluation: evaluate all arguments before
                    // calling the function.
                    LispValue::Function(f) => {
                        instructions.push(Instr::EvalFunctionEager(
                            f,
                            expr_list.len(),
                            is_tail_call,
                        ));
                        instructions.extend(expr_list.into_iter().rev().map(Instr::EvalAndPush));
                    }
                    _ => return Err(EvaluationError::NonFunctionApplication),
                }
            }
            Instr::EvalFunctionEager(func, arg_count, is_tail_call) => {
                match func {
                    LispFunc::BuiltIn(ref n) if n == "list" => {
                        let len = return_values.len();
                        let new_vec = return_values.split_off(len - arg_count);
                        return_values.push(LispValue::SubValue(new_vec));
                    }
                    LispFunc::BuiltIn(ref n) if n == "car" => {
                        if arg_count == 1 {
                            unitary_list(&mut return_values, |mut vec| match vec.pop() {
                                Some(car) => Ok(car),
                                None => Err(EvaluationError::EmptyList),
                            })?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "cdr" => {
                        if arg_count == 1 {
                            unitary_list(&mut return_values, |mut vec| match vec.pop() {
                                Some(_) => Ok(LispValue::SubValue(vec)),
                                None => Err(EvaluationError::EmptyList),
                            })?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "null?" => {
                        if arg_count == 1 {
                            unitary_list(&mut return_values, |vec| {
                                Ok(LispValue::Boolean(vec.is_empty()))
                            })?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "add1" => {
                        if arg_count == 1 {
                            unitary_int(&mut return_values, |i| Ok(LispValue::Integer(i + 1)))?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "sub1" => {
                        if arg_count == 1 {
                            unitary_int(&mut return_values, |i| if i > 0 {
                                Ok(LispValue::Integer(i - 1))
                            } else {
                                Err(EvaluationError::SubZero)
                            })?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "cons" => {
                        if arg_count == 2 {
                            if let LispValue::SubValue(mut new_vec) = return_values.pop().unwrap() {
                                new_vec.push(return_values.pop().unwrap());
                                return_values.push(LispValue::SubValue(new_vec));
                            } else {
                                return Err(EvaluationError::ArgumentTypeMismatch);
                            }
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) if n == "zero?" => {
                        if arg_count == 1 {
                            unitary_int(&mut return_values, |i| Ok(LispValue::Boolean(i == 0)))?
                        } else {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                    }
                    LispFunc::BuiltIn(ref n) => {
                        return Err(EvaluationError::UnknownVariable(n.clone()))
                    }
                    LispFunc::Custom {
                        arg_count: da_arg_count,
                        body,
                    } => {
                        if da_arg_count != arg_count {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }

                        if is_tail_call {
                            // Remove old arguments.
                            // FIXME: this should be done more efficiently
                            let current_pointer = *stack_pointers.last().unwrap();
                            let cnt = return_values.len() - da_arg_count - current_pointer;
                            for _ in 0..cnt {
                                return_values.remove(current_pointer);
                            }
                            instructions.push(Instr::EvalAndPush(*body));
                        } else {
                            instructions.push(Instr::PopState);
                            instructions.push(Instr::EvalAndPush(*body));
                            instructions.push(Instr::SetStackPointer(
                                return_values.len() - da_arg_count,
                            ));
                        }
                    }
                }
            }
            // Pops boolean value off stack, if true, queue evaluation of
            // first expression, else queue evaluation of the second.
            Instr::PopCondPush(true_expr, false_expr) => {
                if let LispValue::Boolean(b) = return_values.pop().unwrap() {
                    let next_instr = if b { true_expr } else { false_expr };
                    instructions.push(Instr::EvalAndPush(next_instr));
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::PopAndSet(var_name) => {
                state.set_variable(&var_name, return_values.pop().unwrap());
                return_values.push(LispValue::SubValue(Vec::new()));
            }
        }
    }

    *init_state = state;
    assert!(stack_pointers == vec![0]);
    assert!(instructions.is_empty());
    assert!(return_values.len() == 1);
    Ok(return_values.pop().unwrap())
}
