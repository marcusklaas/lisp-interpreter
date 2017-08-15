use super::*;
use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

use smallvec::SmallVec;

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
                .map(|x| (x, LispValue::Function(LispFunc::BuiltIn(x))))
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

#[derive(Debug)]
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

    // Removes top n instructions from stack
    PopInstructions(usize),
    // Pops value from stack - removes top n instructions if true
    CondPopInstructions(usize),
    PushValue(LispValue),
    CloneValue(usize),
    PushVariable(String),


    // Integer specialization instructions

    // Increment integer at the top of the int stack
    IntIncrement,
    // Pop int from int stack onto the value stack
    IntPop,
    // Push integer onto the int stack
    IntPush(u64),
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
        LispValue::SubValue(v) => Ok(stack.push(f(v)?)),
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

// TODO: we could pack many more value types
// into 32 bits - but this will do for now. Plus, the semantics
// are nice now
#[derive(Hash, PartialEq, Eq)]
struct ArgTypes {
    repr: SmallVec<[ValueType; 4]>,
}

impl ArgTypes {
    fn from_type_iterator<I: Iterator<Item = ValueType>>(iter: I) -> ArgTypes {
        let mut repr = SmallVec::new();
        repr.extend(iter);

        ArgTypes { repr: repr }
    }
}

enum BuiltIn {
    AddOne,
    SubOne,
    Cond,
    CheckZero,
    Cons,
    CheckNull,
    Car,
    Cdr,
    Define,
    Lambda,
    List,
}

enum SpecializedExpr {
    Function(Rc<Specialization>),
    BuiltIn(BuiltIn),
    Integer(u64),
    Boolean(bool),
}

impl SpecializedExpr {}

struct Specialization {
    body: SpecializedExpr,
    returnType: ValueType,
}

impl Specialization {
    // fn new<BI>(arg_iter: I, body: &LispExpr, specializations: &mut HashMap<(LispExpr, ArgTypes), Specialization>) -> Specialization<'e> {

    // }
}

// TODO: prove that this cannot go into unbounded recursion
fn compile_expr(expr: LispExpr, stack: &[LispValue], state: &State) -> Result<Vec<Instr>, EvaluationError> {
    let mut vek = vec![];

    match expr {
        LispExpr::Argument(offset) => {
            vek.push(Instr::CloneValue(offset));
        }
        LispExpr::Value(v) => {
            vek.push(Instr::PushValue(v));
        }
        LispExpr::OpVar(n) => {
            vek.push(Instr::PushVariable(n));
        }
        LispExpr::Call(mut expr_list, is_tail_call) => {
            // step 1: remove head expression
            let head_expr = expr_list.remove(0);

            // step 2: queue function evaluation with tail
            // FIXME: this is kinda ugly
            match head_expr {
                LispExpr::OpVar(ref n) if n == "cond" => {
                    destructure!(expr_list, [boolean, true_expr, false_expr], {
                        let true_expr_vec = compile_expr(true_expr, stack, state)?;
                        let false_expr_vec = compile_expr(false_expr, stack, state)?;
                        let true_expr_len = true_expr_vec.len();
                        let false_expr_len = false_expr_vec.len();

                        vek.extend(true_expr_vec);
                        vek.push(Instr::PopInstructions(true_expr_len));
                        vek.extend(false_expr_vec);
                        vek.push(Instr::CondPopInstructions(false_expr_len + 1));
                        vek.extend(compile_expr(boolean, stack, state)?);
                    })?
                }
                LispExpr::OpVar(ref n) if n == "lambda" => {
                    destructure!(
                        expr_list,
                        [arg_list, body],
                        if let LispExpr::Call(arg_vec, _is_tail_call) = arg_list {
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
                            let walked_body =
                                body.replace_args(stack);

                            let f = LispFunc::new_custom(args, walked_body, state);

                            vek.push(Instr::PushValue(LispValue::Function(f)));
                        } else {
                            return Err(EvaluationError::ArgumentTypeMismatch);
                        }
                    )?
                }
                LispExpr::OpVar(ref n) if n == "define" => destructure!(
                    expr_list,
                    [var_name, definition],
                    if let LispExpr::OpVar(name) = var_name {
                        vek.push(Instr::PopAndSet(name));
                        vek.extend(compile_expr(definition, stack, state)?);
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    }
                )?,
                // // Eager argument evaluation: evaluate all arguments before
                // // calling the function.
                // LispValue::Function(f) => {
                //     vek.push(Instr::EvalFunctionEager(f, expr_list.len(), is_tail_call));
                //     vek.extend(expr_list.into_iter().rev().map(Instr::EvalAndPush));
                // }
                _ => {
                    vek.push(Instr::EvalFunction(expr_list, is_tail_call));
                    // step 3: queue evaluation of head
                    vek.extend(compile_expr(head_expr, stack, state)?);
                } 
            }
        }
    }

    Ok(vek)
}


pub fn eval<'e>(expr: &'e LispExpr, state: &mut State) -> Result<LispValue, EvaluationError> {
    let mut return_values: Vec<LispValue> = Vec::new();
    let mut int_stack: Vec<u64> = Vec::new();
    let mut instructions = vec![Instr::EvalAndPush(expr.clone())];
    let mut stack_pointers = vec![];
    let mut current_stack = 0;
    // FIXME: maybe this should map to Option<Specialization>, where a None would
    //        signal that the body cannot be specialized to these types. this
    //        would prevent unspecializable combinations from being tried over
    //        and over needlessly.
    //        but this optimization is probably premature at this point.

    // FIXME2: consider having functions manage their own specializations.
    let mut specializations: HashMap<(LispExpr, ArgTypes), Rc<Specialization>> = HashMap::new();

    while let Some(instr) = instructions.pop() {
        match instr {
            Instr::IntIncrement => {
                // This is a specialized instruction. Specialized instructions
                // can make assumptions about the state of the program.
                // For example, this assumes that the integer stack is not empty.
                // It is up to the scheduler of this instruction to verify that
                // this conditions holds.
                let val = int_stack.pop().unwrap();
                int_stack.push(val + 1);
            }
            Instr::IntPop => {
                let val = int_stack.pop().unwrap();
                return_values.push(LispValue::Integer(val));
            }
            Instr::IntPush(i) => {
                int_stack.push(i);
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
            Instr::PushVariable(n) => {
                if let Some(v) = state.get_variable_value(&n) {
                    return_values.push(v);
                } else {
                    return Err(EvaluationError::UnknownVariable(n));
                }
            }


            Instr::SetStackPointer(p) => {
                stack_pointers.push(current_stack);
                current_stack = p;
            }
            Instr::PopState => {
                let val = return_values.pop().unwrap();

                return_values.truncate(current_stack);
                return_values.push(val);

                current_stack = stack_pointers.pop().unwrap();
            }
            Instr::EvalAndPush(expr) => {
                instructions.extend(compile_expr(expr, &return_values[current_stack..], state)?);
            }
            // Pops a function off the value stack and applies it
            Instr::EvalFunction(expr_list, is_tail_call) => {
                match return_values.pop().unwrap() {
                    LispValue::Function(LispFunc::BuiltIn("cond")) => {
                        destructure!(expr_list, [boolean, true_expr, false_expr], {
                            let true_expr_vec = compile_expr(true_expr, &return_values[current_stack..], state)?;
                            let false_expr_vec = compile_expr(false_expr, &return_values[current_stack..], state)?;
                            let true_expr_len = true_expr_vec.len();
                            let false_expr_len = false_expr_vec.len();

                            instructions.extend(true_expr_vec);
                            instructions.push(Instr::PopInstructions(true_expr_len));
                            instructions.extend(false_expr_vec);
                            instructions.push(Instr::CondPopInstructions(false_expr_len + 1));
                            instructions.extend(compile_expr(boolean, &return_values[current_stack..], state)?);
                            // instructions.push(Instr::EvalAndPush(boolean));
                        })?
                    }
                    LispValue::Function(LispFunc::BuiltIn("lambda")) => {
                        destructure!(
                            expr_list,
                            [arg_list, body],
                            if let LispExpr::Call(arg_vec, _is_tail_call) = arg_list {
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
                                let walked_body =
                                    body.replace_args(&return_values[current_stack..]);

                                let f = LispFunc::new_custom(args, walked_body, state);

                                return_values.push(LispValue::Function(f));
                            } else {
                                return Err(EvaluationError::ArgumentTypeMismatch);
                            }
                        )?
                    }
                    LispValue::Function(LispFunc::BuiltIn("define")) => destructure!(
                        expr_list,
                        [var_name, definition],
                        if let LispExpr::OpVar(name) = var_name {
                            instructions.push(Instr::PopAndSet(name));
                            instructions.push(Instr::EvalAndPush(definition));
                        } else {
                            return Err(EvaluationError::ArgumentTypeMismatch);
                        }
                    )?,
                    // Eager argument evaluation: evaluate all arguments before
                    // calling the function.
                    LispValue::Function(f) => {
                        instructions
                            .push(Instr::EvalFunctionEager(f, expr_list.len(), is_tail_call));
                        instructions.extend(expr_list.into_iter().rev().map(Instr::EvalAndPush));
                    }
                    _ => return Err(EvaluationError::NonFunctionApplication),
                }
            }
            Instr::EvalFunctionEager(func, arg_count, is_tail_call) => {
                match func {
                    LispFunc::BuiltIn("list") => {
                        let len = return_values.len();
                        let new_vec = return_values.split_off(len - arg_count);
                        return_values.push(LispValue::SubValue(new_vec));
                    }
                    LispFunc::BuiltIn("car") => if arg_count == 1 {
                        unitary_list(&mut return_values, |mut vec| match vec.pop() {
                            Some(car) => Ok(car),
                            None => Err(EvaluationError::EmptyList),
                        })?
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("cdr") => if arg_count == 1 {
                        unitary_list(&mut return_values, |mut vec| match vec.pop() {
                            Some(_) => Ok(LispValue::SubValue(vec)),
                            None => Err(EvaluationError::EmptyList),
                        })?
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("null?") => if arg_count == 1 {
                        unitary_list(&mut return_values, |vec| {
                            Ok(LispValue::Boolean(vec.is_empty()))
                        })?
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("add1") => if arg_count == 1 {
                        if let LispValue::Integer(i) = return_values.pop().unwrap() {
                            instructions.push(Instr::IntPop);
                            instructions.push(Instr::IntIncrement);
                            instructions.push(Instr::IntPush(i));
                        } else {
                            return Err(EvaluationError::ArgumentTypeMismatch);
                        }
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("sub1") => if arg_count == 1 {
                        unitary_int(&mut return_values, |i| if i > 0 {
                            Ok(LispValue::Integer(i - 1))
                        } else {
                            Err(EvaluationError::SubZero)
                        })?
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("cons") => if arg_count == 2 {
                        if let LispValue::SubValue(mut new_vec) = return_values.pop().unwrap() {
                            new_vec.push(return_values.pop().unwrap());
                            return_values.push(LispValue::SubValue(new_vec));
                        } else {
                            return Err(EvaluationError::ArgumentTypeMismatch);
                        }
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn("zero?") => if arg_count == 1 {
                        unitary_int(&mut return_values, |i| Ok(LispValue::Boolean(i == 0)))?
                    } else {
                        return Err(EvaluationError::ArgumentCountMismatch);
                    },
                    LispFunc::BuiltIn(n) => return Err(EvaluationError::UnknownVariable(n.into())),
                    LispFunc::Custom {
                        arg_count: da_arg_count,
                        body,
                    } => {
                        // Too many arguments or none at all.
                        if da_arg_count < arg_count || arg_count == 0 {
                            return Err(EvaluationError::ArgumentCountMismatch);
                        }
                        // Not enough arguments, let's create a lambda that takes
                        // the remainder.
                        else if arg_count < da_arg_count {
                            let orig_func = LispFunc::Custom {
                                arg_count: da_arg_count,
                                body: body,
                            };
                            let continuation = LispFunc::create_continuation(
                                orig_func,
                                da_arg_count,
                                arg_count,
                                &return_values[current_stack..],
                            );
                            return_values.truncate(current_stack);
                            return_values.push(LispValue::Function(continuation));
                        }
                        // Exactly right number of arguments. Let's evaluate.
                        else if is_tail_call {
                            // Remove old arguments of the stack.
                            let top_index = return_values.len() - arg_count;
                            return_values.splice(current_stack..top_index, iter::empty());

                            instructions.extend(compile_expr(*body, &return_values[current_stack..], state)?);
                            // instructions.push(Instr::EvalAndPush(*body));
                        } else {
                            instructions.push(Instr::PopState);
                            instructions.extend(compile_expr(*body, &return_values[current_stack..], state)?);
                            // instructions.push(Instr::EvalAndPush(*body));
                            instructions
                                .push(Instr::SetStackPointer(return_values.len() - arg_count));
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

    assert!(stack_pointers.is_empty());
    assert_eq!(current_stack, 0);
    assert!(instructions.is_empty());
    assert_eq!(return_values.len(), 1);
    Ok(return_values.pop().unwrap())
}
