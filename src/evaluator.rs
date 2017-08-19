use super::*;
use std::collections::HashMap;
use std::iter;
use std::mem::swap;

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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Instr {
    /// Calls the function we're currently in with the given number of arguments
    /// at the top of the stack
    Recurse(usize),
    /// Evaluates the function at the top of the stack with given number of arguments.
    /// The booleans indicates whether this is a tail call
    EvalFunction(usize, bool),
    /// Pops a value from the stack and adds it to the state at the given name
    PopAndSet(String),
    /// Creates a custom function with given (arguments, function body) and pushes
    /// the result to the stack
    CreateLambda(Vec<LispExpr>, LispExpr),

    /// Skips the given number of instructions
    Jump(usize),
    /// Pops boolean value from stack and conditionally jumps a number of instructions
    CondJump(usize),
    /// Pushes a value to the stack
    PushValue(LispValue),
    /// Clones the n'th argument to the function and pushes it to the stack
    CloneArgument(usize),
    /// Clones value from state at given index and pushes it to the stack
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

fn unitary_int<F: Fn(u64) -> EvaluationResult<LispValue>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> EvaluationResult<()> {
    let reference = stack.last_mut().unwrap();

    if let &mut LispValue::Integer(i) = reference {
        Ok(*reference = f(i)?)
    } else {
        Err(EvaluationError::ArgumentTypeMismatch)
    }
}

fn unitary_list<F: Fn(Vec<LispValue>) -> EvaluationResult<LispValue>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> EvaluationResult<()> {
    match stack.pop().unwrap() {
        LispValue::List(v) => Ok(stack.push(f(v)?)),
        _ => Err(EvaluationError::ArgumentTypeMismatch),
    }
}

macro_rules! destructure {
    ( $y:ident, $x:expr ) => {{$x}};

    ( $y:ident, [ $( $i:ident ),* ], $body:expr ) => {
        {
            if let ($( Some( $i), )* None) = {
                let mut iter = $y.into_iter();
                ( $( destructure!($i, iter.next()), )* iter.next() )
            } {
                Ok($body)
            } else {
                Err(EvaluationError::ArgumentCountMismatch)
            }?
        }
    };
}

pub fn compile_expr(expr: LispExpr, state: &State) -> EvaluationResult<Vec<Instr>> {
    let mut vek = vec![];

    match expr {
        LispExpr::Argument(offset) => {
            vek.push(Instr::CloneArgument(offset));
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
        LispExpr::Call(mut expr_list, is_tail_call, is_self_call) => {
            let head_expr = expr_list.remove(0);

            match head_expr {
                LispExpr::Macro(LispMacro::Cond) => {
                    destructure!(expr_list, [boolean, true_expr, false_expr], {
                        let true_expr_vec = compile_expr(true_expr, state)?;
                        let false_expr_vec = compile_expr(false_expr, state)?;
                        let true_expr_len = true_expr_vec.len();
                        let false_expr_len = false_expr_vec.len();

                        vek.extend(true_expr_vec);
                        vek.push(Instr::Jump(true_expr_len));
                        vek.extend(false_expr_vec);
                        vek.push(Instr::CondJump(false_expr_len + 1));
                        vek.extend(compile_expr(boolean, state)?);
                    })
                }
                LispExpr::Macro(LispMacro::Lambda) => destructure!(
                    expr_list,
                    [arg_list, body],
                    if let LispExpr::Call(arg_vec, _is_tail_call, _is_self_call) = arg_list {
                        vek.push(Instr::CreateLambda(arg_vec, body));
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    }
                ),
                LispExpr::Macro(LispMacro::Define) => destructure!(
                    expr_list,
                    [var_name, definition],
                    if let LispExpr::OpVar(name) = var_name {
                        let mut def = definition;
                        def.flag_self_calls(&name);
                        vek.push(Instr::PopAndSet(name.clone()));
                        vek.extend(compile_expr(def, state)?);
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    }
                ),
                LispExpr::Value(LispValue::Function(LispFunc::BuiltIn(f))) => {
                    let instr = builtin_instr(f, expr_list.len())?;
                    vek.push(instr);

                    for expr in expr_list.into_iter().rev() {
                        let instr_vec = compile_expr(expr, state)?;
                        vek.extend(instr_vec);
                    }
                }
                _ => {
                    if is_tail_call && is_self_call {
                        vek.push(Instr::Recurse(expr_list.len()));
                    } else {
                        vek.push(Instr::EvalFunction(expr_list.len(), is_tail_call));
                        vek.extend(compile_expr(head_expr, state)?);
                    }

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

fn builtin_instr(f: BuiltIn, arg_count: usize) -> EvaluationResult<Instr> {
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

#[derive(Clone, Debug)]
struct StackRef {
    instr_pointer: usize,
    // TODO: this should probably be a UnsafeCell later. we set it only once
    // and never update later
    instr_vec: Rc<RefCell<Vec<Instr>>>,
    stack_pointer: usize,
}

pub fn eval<'e>(expr: &'e LispExpr, state: &mut State) -> EvaluationResult<LispValue> {
    let mut return_values: Vec<LispValue> = Vec::new();
    let initial_instructions = compile_expr(expr.clone(), state)?;
    let mut stax = vec![];
    let mut stack_ref = StackRef {
        instr_pointer: initial_instructions.len(),
        instr_vec: Rc::new(RefCell::new(initial_instructions)),
        stack_pointer: 0,
    };

    'l: loop {
        // Pop stack frame when there's no more instructions in this one
        while stack_ref.instr_pointer == 0 {
            let val = return_values.pop().unwrap();
            return_values.truncate(stack_ref.stack_pointer);
            return_values.push(val);

            if let Some(new_stack_ref) = stax.pop() {
                stack_ref = new_stack_ref;
            } else {
                break 'l;
            }
        }

        let mut update_stacks = None;

        {
            stack_ref.instr_pointer -= 1;
            let instr = &stack_ref.instr_vec.borrow()[stack_ref.instr_pointer];

            match *instr {
                Instr::Recurse(arg_count) => {
                    let top_index = return_values.len() - arg_count;
                    return_values.splice(stack_ref.stack_pointer..top_index, iter::empty());
                    stack_ref.instr_pointer = { stack_ref.instr_vec.borrow().len() };
                }
                Instr::CreateLambda(ref arg_vec, ref body) => {
                    let args = arg_vec
                        .into_iter()
                        .map(|expr| match *expr {
                            LispExpr::OpVar(ref name) => Ok(&name[..]),
                            _ => Err(EvaluationError::MalformedDefinition),
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    // If there are any references to function arguments in
                    // the lambda body, we should resolve them before
                    // creating the lambda.
                    // This enables us to do closures.
                    let walked_body = body.replace_args(&return_values[stack_ref.stack_pointer..]);
                    let f = LispFunc::new_custom(&args, walked_body, state);

                    return_values.push(LispValue::Function(f));
                }
                Instr::Jump(n) => {
                    stack_ref.instr_pointer -= n;
                }
                Instr::CondJump(n) => if let LispValue::Boolean(b) = return_values.pop().unwrap() {
                    if b {
                        stack_ref.instr_pointer -= n;
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                },
                Instr::PushValue(ref v) => {
                    return_values.push(v.clone());
                }
                Instr::CloneArgument(offset) => {
                    let index = stack_ref.stack_pointer + offset;
                    let value = return_values[index].clone();
                    return_values.push(value);
                }
                Instr::PushVariable(i) => {
                    return_values.push(state.get(i));
                }
                Instr::PopAndSet(ref var_name) => {
                    state.set_variable(var_name, return_values.pop().unwrap());
                    return_values.push(LispValue::List(Vec::new()));
                }
                // Pops a function off the value stack and applies it
                Instr::EvalFunction(arg_count, is_tail_call) => {
                    if let LispValue::Function(funk) = return_values.pop().unwrap() {
                        match funk {
                            LispFunc::BuiltIn(b) => {
                                // The performance of this solution is basically horrendous,
                                // but all the performant solutions are super messy.
                                // This shouldn't occur too often, though.
                                let instr_vec =
                                    Rc::new(RefCell::new(vec![builtin_instr(b, arg_count)?]));
                                update_stacks = Some((instr_vec, arg_count, true));
                            }
                            LispFunc::Custom(f) => {
                                // Too many arguments or none at all.
                                if f.arg_count < arg_count || arg_count == 0 {
                                    return Err(EvaluationError::ArgumentCountMismatch);
                                }
                                // Not enough arguments, let's create a lambda that takes
                                // the remainder.
                                else if arg_count < f.arg_count {
                                    let temp_stack = return_values.len() - arg_count;
                                    let f_arg_count = f.arg_count;
                                    let continuation = LispFunc::create_continuation(
                                        f,
                                        f_arg_count,
                                        arg_count,
                                        &return_values[temp_stack..],
                                    );

                                    return_values.truncate(temp_stack);
                                    return_values.push(LispValue::Function(continuation));
                                }
                                // Exactly right number of arguments. Let's evaluate.
                                else if is_tail_call {
                                    // Remove old arguments of the stack.
                                    let top_index = return_values.len() - arg_count;
                                    return_values
                                        .splice(stack_ref.stack_pointer..top_index, iter::empty());

                                    update_stacks = Some((f.compile(state)?, arg_count, false));
                                } else {
                                    update_stacks = Some((f.compile(state)?, arg_count, true));
                                }
                            }
                        }
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
                Instr::AddOne => {
                    unitary_int(&mut return_values, |i| Ok(LispValue::Integer(i + 1)))?
                }
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
            }
        }

        // This solution is not very elegant, but it's necessary
        // to please the borrowchecker in a safe manner.
        if let Some((next_instr_vec, arg_count, push_stack)) = update_stacks {
            let instr_len = { next_instr_vec.borrow().len() };
            let mut next_stack_ref = StackRef {
                instr_pointer: instr_len,
                instr_vec: next_instr_vec,
                stack_pointer: return_values.len() - arg_count,
            };
            swap(&mut next_stack_ref, &mut stack_ref);

            if push_stack {
                stax.push(next_stack_ref);
            }
        }
    }

    assert!(stax.is_empty());
    assert_eq!(stack_ref.instr_pointer, 0);
    assert_eq!(return_values.len(), 1);
    Ok(return_values.pop().unwrap())
}
