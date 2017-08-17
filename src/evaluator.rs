use super::*;
use std::collections::HashMap;
use std::iter;

#[derive(Debug, Clone)]
pub struct State {
    index_map: HashMap<String, usize>,
    store: Vec<LispValue>,
}

impl State {
    pub fn new() -> State {
        State {
            index_map: HashMap::new(),
            store: Vec::new(),
        }
    }

    // FIXME: this store business may be totally unnecessary
    pub fn get_index(&self, var_name: &str) -> Option<usize> {
        self.index_map.get(var_name).cloned()
    }

    pub fn get(&self, index: usize) -> LispValue {
        self.store[index].clone()
    }

    pub fn get_variable_value(&self, var_name: &str) -> Option<LispValue> {
        self.index_map
            .get(var_name)
            .map(|&index| self.store[index].clone())
    }

    pub fn set_variable(&mut self, var_name: &str, val: LispValue) {
        let index = self.store.len();
        self.index_map.insert(var_name.into(), index);
        self.store.push(val);
    }
}

// TODO: remove superfluous derives
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Instr {
    // Function and argument vector, Tail call
    EvalFunction(usize, bool),
    PopAndSet(String),
    PopState,
    // Function, Argument Count, Tail call
    EvalFunctionEager(LispFunc, usize, bool),
    // Argument List, LispExpr
    CreateLambda(Vec<LispExpr>, LispExpr),

    // Removes top n instructions from stack
    PopInstructions(usize),
    // Pops value from stack - removes top n instructions if true
    CondPopInstructions(usize),
    PushValue(LispValue),
    CloneValue(usize),
    // Index in state
    PushVariable(usize),

    // Built-in instructions
    AddOne,
    SubOne,
    Cons,
    Cdr,
    Car,
    List(usize),
    CheckZero,
    CheckNull,
}

fn unitary_int<F: Fn(u64) -> Result<LispValue, EvaluationError>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> Result<(), EvaluationError> {
    match stack.pop().unwrap() {
        LispValue::Integer(i) => Ok(stack.push(f(i)?)),
        _ => Err(EvaluationError::ArgumentTypeMismatch),
    }
}

fn unitary_list<F: Fn(Vec<LispValue>) -> Result<LispValue, EvaluationError>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> Result<(), EvaluationError> {
    match stack.pop().unwrap() {
        LispValue::List(v) => Ok(stack.push(f(v)?)),
        _ => Err(EvaluationError::ArgumentTypeMismatch),
    }
}

macro_rules! destructure {
    ( $y:ident, $x:expr ) => {{$x}};

    ( $y:ident, [ $( $i:ident ),* ], $body:expr ) => {
        {
            if let ($( Some($i), )* None) = {
                let mut iter = $y.into_iter();
                ( $( destructure!($i, iter.next()), )* iter.next() )
            } {
                Ok($body)
            } else {
                Err(EvaluationError::ArgumentCountMismatch)
            }
        }
    };
}

pub fn compile_expr(expr: LispExpr, state: &State) -> Result<Vec<Instr>, EvaluationError> {
    let mut vek = vec![];

    match expr {
        LispExpr::Argument(offset) => {
            vek.push(Instr::CloneValue(offset));
        }
        LispExpr::Value(v) => {
            vek.push(Instr::PushValue(v));
        }
        LispExpr::OpVar(n) => if let Some(i) = state.get_index(&n) {
            vek.push(Instr::PushVariable(i));
        } else {
            return Err(EvaluationError::UnknownVariable(n));
        },
        LispExpr::Macro(..) => {
            return Err(EvaluationError::UnexpectedOperator);
        }
        LispExpr::Call(mut expr_list, is_tail_call) => {
            let head_expr = expr_list.remove(0);

            match head_expr {
                LispExpr::Macro(LispMacro::Cond) => {
                    destructure!(expr_list, [boolean, true_expr, false_expr], {
                        let true_expr_vec = compile_expr(true_expr, state)?;
                        let false_expr_vec = compile_expr(false_expr, state)?;
                        let true_expr_len = true_expr_vec.len();
                        let false_expr_len = false_expr_vec.len();

                        vek.extend(true_expr_vec);
                        vek.push(Instr::PopInstructions(true_expr_len));
                        vek.extend(false_expr_vec);
                        vek.push(Instr::CondPopInstructions(false_expr_len + 1));
                        vek.extend(compile_expr(boolean, state)?);
                    })?
                }
                LispExpr::Macro(LispMacro::Lambda) => destructure!(
                    expr_list,
                    [arg_list, body],
                    if let LispExpr::Call(arg_vec, _is_tail_call) = arg_list {
                        vek.push(Instr::CreateLambda(arg_vec, body));
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    }
                )?,
                LispExpr::Macro(LispMacro::Define) => destructure!(
                    expr_list,
                    [var_name, definition],
                    if let LispExpr::OpVar(name) = var_name {
                        vek.push(Instr::PopAndSet(name));
                        vek.extend(compile_expr(definition, state)?);
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    }
                )?,
                LispExpr::Value(LispValue::Function(LispFunc::BuiltIn(f))) => {
                    let instr = builtin_instr(f, expr_list.len())?;
                    vek.push(instr);

                    for expr in expr_list.into_iter().rev() {
                        let instr_vec = compile_expr(expr, state)?;
                        vek.extend(instr_vec);
                    }
                }
                _ => {
                    vek.push(Instr::EvalFunction(expr_list.len(), is_tail_call));
                    vek.extend(compile_expr(head_expr, state)?);

                    for expr in expr_list.into_iter().rev() {
                        let instr_vec = compile_expr(expr, state)?;
                        vek.extend(instr_vec);
                    }
                }
            }
        }
    }

    Ok(vek)
}

fn builtin_instr(f: BuiltIn, arg_count: usize) -> Result<Instr, EvaluationError> {
    Ok(match (f, arg_count) {
        (BuiltIn::AddOne, 1) => Instr::AddOne,
        (BuiltIn::SubOne, 1) => Instr::SubOne,
        (BuiltIn::CheckZero, 1) => Instr::CheckZero,
        (BuiltIn::CheckNull, 1) => Instr::CheckNull,
        (BuiltIn::List, _) => Instr::List(arg_count),
        (BuiltIn::Cons, 2) => Instr::Cons,
        (BuiltIn::Car, 1) => Instr::Car,
        (BuiltIn::Cdr, 1) => Instr::Cdr,
        (_, _) => return Err(EvaluationError::ArgumentCountMismatch),
    })
}

pub fn eval<'e>(expr: &'e LispExpr, state: &mut State) -> Result<LispValue, EvaluationError> {
    let mut return_values: Vec<LispValue> = Vec::new();
    let mut instructions = compile_expr(expr.clone(), state)?;
    let mut stack_pointers = vec![];
    let mut current_stack = 0;

    while let Some(instr) = instructions.pop() {
        match instr {
            Instr::CreateLambda(arg_vec, body) => {
                let args = arg_vec
                    .into_iter()
                    .map(|expr| match expr {
                        LispExpr::OpVar(name) => Ok(name),
                        _ => Err(EvaluationError::MalformedDefinition),
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // If there are any references to function arguments in
                // the lambda body, we should resolve them before
                // creating the lambda.
                // This enables us to do closures.
                let walked_body = body.replace_args(&return_values[current_stack..]);
                let f = LispFunc::new_custom(args, walked_body, state);

                return_values.push(LispValue::Function(f));
            }
            Instr::PopInstructions(n) => {
                let new_len = instructions.len() - n;
                instructions.truncate(new_len);
            }
            Instr::CondPopInstructions(n) => {
                if let LispValue::Boolean(b) = return_values.pop().unwrap() {
                    if b {
                        instructions.push(Instr::PopInstructions(n));
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::PushValue(v) => {
                return_values.push(v);
            }
            Instr::CloneValue(offset) => {
                let index = current_stack + offset;
                let value = return_values[index].clone();
                return_values.push(value);
            }
            Instr::PushVariable(i) => {
                return_values.push(state.get(i));
            }
            Instr::PopState => {
                let val = return_values.pop().unwrap();

                return_values.truncate(current_stack);
                return_values.push(val);

                current_stack = stack_pointers.pop().unwrap();
            }
            // Pops a function off the value stack and applies it
            // TODO: we do not really need this any more! remove it!
            Instr::EvalFunction(expr_list_len, is_tail_call) => {
                if let LispValue::Function(f) = return_values.pop().unwrap() {
                    instructions.push(Instr::EvalFunctionEager(f, expr_list_len, is_tail_call));
                } else {
                    return Err(EvaluationError::NonFunctionApplication);
                }
            }
            Instr::List(arg_count) => {
                let len = return_values.len();
                let new_vec = return_values.split_off(len - arg_count);
                return_values.push(LispValue::List(new_vec));
            }
            Instr::Car => unitary_list(&mut return_values, |mut vec| match vec.pop() {
                Some(car) => Ok(car),
                None => Err(EvaluationError::EmptyList),
            })?,
            Instr::Cdr => unitary_list(&mut return_values, |mut vec| match vec.pop() {
                Some(_) => Ok(LispValue::List(vec)),
                None => Err(EvaluationError::EmptyList),
            })?,
            Instr::CheckNull => unitary_list(&mut return_values, |vec| {
                Ok(LispValue::Boolean(vec.is_empty()))
            })?,
            Instr::AddOne => unitary_int(&mut return_values, |i| Ok(LispValue::Integer(i + 1)))?,
            Instr::SubOne => unitary_int(&mut return_values, |i| if i > 0 {
                Ok(LispValue::Integer(i - 1))
            } else {
                Err(EvaluationError::SubZero)
            })?,
            Instr::Cons => if let LispValue::List(mut new_vec) = return_values.pop().unwrap() {
                new_vec.push(return_values.pop().unwrap());
                return_values.push(LispValue::List(new_vec));
            } else {
                return Err(EvaluationError::ArgumentTypeMismatch);
            },
            Instr::CheckZero => {
                unitary_int(&mut return_values, |i| Ok(LispValue::Boolean(i == 0)))?
            }
            Instr::EvalFunctionEager(funk, arg_count, is_tail_call) => {
                match funk {
                    LispFunc::BuiltIn(b) => instructions.push(builtin_instr(b, arg_count)?),
                    LispFunc::Custom(mut f) => {
                        // Too many arguments or none at all.
                        if f.arg_count < arg_count || arg_count == 0 {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                        // Not enough arguments, let's create a lambda that takes
                        // the remainder.
                        else if arg_count < f.arg_count {
                            stack_pointers.push(current_stack);
                            current_stack = return_values.len() - arg_count;

                            let funk_arg_count = f.arg_count;
                            let continuation = LispFunc::create_continuation(
                                LispFunc::Custom(f),
                                funk_arg_count,
                                arg_count,
                                &return_values[current_stack..],
                            );

                            instructions.push(Instr::PopState);
                            return_values.truncate(current_stack);
                            return_values.push(LispValue::Function(continuation));
                        }
                        // Exactly right number of arguments. Let's evaluate.
                        else if is_tail_call {
                            // Remove old arguments of the stack.
                            let top_index = return_values.len() - arg_count;
                            return_values.splice(current_stack..top_index, iter::empty());
                            instructions.extend(f.compile(state)?);
                        } else {
                            stack_pointers.push(current_stack);
                            current_stack = return_values.len() - arg_count;
                            instructions.push(Instr::PopState);
                            instructions.extend(f.compile(state)?);
                        }
                    }
                }
            }
            Instr::PopAndSet(var_name) => {
                state.set_variable(&var_name, return_values.pop().unwrap());
                return_values.push(LispValue::List(Vec::new()));
            }
        }
    }

    assert!(stack_pointers.is_empty());
    assert_eq!(current_stack, 0);
    assert!(instructions.is_empty());
    assert_eq!(return_values.len(), 1);
    Ok(return_values.pop().unwrap())
}
