use super::{ArgType, BuiltIn, CustomFunc, EvaluationError, EvaluationResult, FinalizationContext,
            FinalizedExpr, LispExpr, LispFunc, LispValue, TopExpr};
use super::specialization;
use std::collections::hash_map;
use std::collections::HashMap;
use std::iter;
use std::ops::Index;
use std::mem::{swap, transmute};
use std::default::Default;

// TODO: ideally, this shouldn't be public - so
// we can guarantee that no invalid indices can be
// constructed
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct StateIndex(usize);

#[derive(Debug, Clone)]
pub struct State {
    index_map: HashMap<String, usize>,
    store: Vec<LispValue>,
}

impl Index<StateIndex> for State {
    type Output = LispValue;

    fn index(&self, index: StateIndex) -> &LispValue {
        self.store.index(index.0)
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            index_map: HashMap::new(),
            store: Vec::new(),
        }
    }
}

impl State {
    pub fn get_index(&self, var_name: &str) -> Option<StateIndex> {
        self.index_map.get(var_name).map(|&i| StateIndex(i))
    }

    pub fn set_variable(
        &mut self,
        var_name: String,
        val: LispValue,
        allow_override: bool,
    ) -> EvaluationResult<()> {
        let entry = self.index_map.entry(var_name);

        if let hash_map::Entry::Occupied(occ_entry) = entry {
            if !allow_override {
                return Err(EvaluationError::BadDefine);
            } else {
                let index = *occ_entry.get();
                self.store[index] = val;
            }
        } else {
            let index = self.store.len();
            entry.or_insert(index);
            self.store.push(val);
        }

        Ok(())
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
    /// Creates a custom function with given (scope level, argument count, function body) and pushes
    /// the result to the stack
    CreateLambda(usize, usize, FinalizedExpr),

    /// Skips the given number of instructions
    Jump(usize),
    /// Pops boolean value from stack and conditionally jumps a number of instructions
    CondJump(usize),
    /// Pushes a value to the stack
    PushValue(LispValue),
    /// Clones the n'th argument to the function and pushes it to the stack
    CloneArgument(usize),
    /// Moves the n'th argument to the function to the top of the stack and replaces
    /// it with a dummy value.
    MoveArgument(usize),
    /// Clones value from state at given index and pushes it to the stack
    PushVariable(StateIndex),

    // Built-in instructions
    AddOne,
    SubOne,
    Cons,
    Cdr,
    Car,
    List(usize),
    CheckZero,
    CheckNull,
    CheckType(ArgType),
}

fn unitary_int<F: Fn(u64) -> EvaluationResult<LispValue>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> EvaluationResult<()> {
    let reference = stack.last_mut().unwrap();

    if let LispValue::Integer(i) = *reference {
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

#[macro_export]
macro_rules! destructure {
    ( $y:ident, $x:expr ) => {{$x}};

    ( $iter:ident, [ $( $i:ident ),* ], $body:expr ) => {
        {
            if let ($( Some( $i), )* None) = {
                ( $( destructure!($i, $iter.next()), )* $iter.next() )
            } {
                Ok($body)
            } else {
                Err(EvaluationError::ArgumentCountMismatch)
            }?
        }
    };
}

fn compile_top_expr(expr: TopExpr, state: &State) -> EvaluationResult<Vec<Instr>> {
    match expr {
        TopExpr::Define(name, sub_expr) => {
            let finalized_definition =
                sub_expr.finalize(&mut FinalizationContext::new(Some(&name)))?;

            let mut instructions = vec![Instr::PopAndSet(name)];
            instructions.extend(compile_finalized_expr(finalized_definition, state)?);
            Ok(instructions)
        }
        TopExpr::Regular(sub_expr) => compile_finalized_expr(sub_expr, state),
    }
}

pub fn compile_finalized_expr(expr: FinalizedExpr, state: &State) -> EvaluationResult<Vec<Instr>> {
    let mut instructions = Vec::new();

    match expr {
        FinalizedExpr::Argument(offset, _scope, true) => {
            instructions.push(Instr::MoveArgument(offset));
        }
        FinalizedExpr::Argument(offset, _scope, false) => {
            instructions.push(Instr::CloneArgument(offset));
        }
        FinalizedExpr::Value(v) => {
            instructions.push(Instr::PushValue(v));
        }
        FinalizedExpr::Variable(n) => if let Some(i) = state.get_index(&n) {
            instructions.push(Instr::PushVariable(i));
        } else {
            return Err(EvaluationError::UnknownVariable(n));
        },
        FinalizedExpr::Cond(triple) => {
            let unpacked = *triple;
            let (test, true_expr, false_expr) = unpacked;
            let true_expr_vec = compile_finalized_expr(true_expr, state)?;
            let false_expr_vec = compile_finalized_expr(false_expr, state)?;
            let true_expr_len = true_expr_vec.len();
            let false_expr_len = false_expr_vec.len();

            instructions.extend(true_expr_vec);
            instructions.push(Instr::Jump(true_expr_len));
            instructions.extend(false_expr_vec);
            instructions.push(Instr::CondJump(false_expr_len + 1));
            instructions.extend(compile_finalized_expr(test, state)?);
        }
        FinalizedExpr::Lambda(arg_count, scope, body) => {
            instructions.push(Instr::CreateLambda(arg_count, scope, *body));
        }
        FinalizedExpr::FunctionCall(funk, args, is_tail_call, is_self_call) => {
            if let FinalizedExpr::Value(LispValue::Function(LispFunc::BuiltIn(f))) = *funk {
                instructions.push(builtin_instr(f, args.len())?);
            } else if is_tail_call && is_self_call {
                instructions.push(Instr::Recurse(args.len()));
            } else {
                instructions.push(Instr::EvalFunction(args.len(), is_tail_call));
                instructions.extend(compile_finalized_expr(*funk, state)?);
            }

            for expr in args.into_iter().rev() {
                let instr_vec = compile_finalized_expr(expr, state)?;
                instructions.extend(instr_vec);
            }
        }
    }

    Ok(instructions)
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
        (BuiltIn::CheckType(t), 1) => Instr::CheckType(t),
        (_, _) => return Err(EvaluationError::ArgumentCountMismatch),
    })
}

struct StackRef {
    instr_pointer: usize,
    #[allow(dead_code)] func: CustomFunc,
    stack_pointer: usize,
    // This reference isn't really static - it refers to vector inside of
    // instr_vec. There's just no way to express this in Rust (I think!)
    instr_slice: &'static [Instr],
}

impl StackRef {
    fn new(func: CustomFunc, stack_pointer: usize, state: &State) -> EvaluationResult<StackRef> {
        let reference = unsafe { transmute(func.compile(state)?) };

        Ok(StackRef {
            instr_slice: reference,
            instr_pointer: reference.len(),
            func: func,
            stack_pointer: stack_pointer,
        })
    }
}

pub fn eval(expr: LispExpr, state: &mut State) -> EvaluationResult<LispValue> {
    let mut return_values: Vec<LispValue> = Vec::new();
    let mut stax = vec![];
    let mut stack_ref = {
        let top_expr = expr.into_top_expr()?;
        let instructions = compile_top_expr(top_expr, state)?;
        let main_func = CustomFunc::from_byte_code(0, instructions);
        StackRef::new(main_func, 0, state)?
    };

    'l: loop {
        // Pop stack frame when there's no more instructions in this one
        while stack_ref.instr_pointer == 0 {
            let top_index = return_values.len() - 1;
            return_values.splice(stack_ref.stack_pointer..top_index, iter::empty());

            if let Some(new_stack_ref) = stax.pop() {
                stack_ref = new_stack_ref;
            } else {
                break 'l;
            }
        }

        let mut update_stacks = None;
        stack_ref.instr_pointer -= 1;

        match stack_ref.instr_slice[stack_ref.instr_pointer] {
            Instr::Recurse(arg_count) => {
                let top_index = return_values.len() - arg_count;
                return_values.splice(stack_ref.stack_pointer..top_index, iter::empty());
                stack_ref.instr_pointer = { stack_ref.instr_slice.len() };
            }
            Instr::CreateLambda(arg_count, scope, ref body) => {
                // If there are any references to function arguments in
                // the lambda body, we should resolve them before
                // creating the lambda.
                // This enables us to do closures.
                let walked_body =
                    body.replace_args(scope, &mut return_values[stack_ref.stack_pointer..]);
                let f = LispFunc::new_custom(arg_count, walked_body);

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
            Instr::MoveArgument(offset) => {
                let len = return_values.len();
                return_values.push(LispValue::Boolean(false));
                return_values.swap(stack_ref.stack_pointer + offset, len);
            }
            Instr::PushVariable(i) => {
                return_values.push(state[i].clone());
            }
            Instr::PopAndSet(ref var_name) => {
                state.set_variable(var_name.clone(), return_values.pop().unwrap(), false)?;
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
                            let func = CustomFunc::from_byte_code(
                                arg_count,
                                vec![builtin_instr(b, arg_count)?],
                            );

                            update_stacks = Some((func, true));
                        }
                        LispFunc::Custom(f) => {
                            // // FIXME: DEBUGGING ONLY
                            // let index = return_values.len() - arg_count;
                            // let arg_types = return_values[index..]
                            //     .iter()
                            //     .map(LispValue::get_type)
                            //     .collect::<Vec<_>>();
                            // if specialization::can_specialize(&f, &arg_types, state) {
                            //     // TODO: we can specialize! do something with it
                            //     println!("specialization succeeded");
                            // } else {
                            //     println!("specialization failed");
                            // }

                            let func_arg_count = f.0.arg_count;

                            // Too many arguments or none at all.
                            if func_arg_count < arg_count || arg_count == 0 {
                                return Err(EvaluationError::ArgumentCountMismatch);
                            }
                            // Not enough arguments, let's create a lambda that takes
                            // the remainder.
                            else if arg_count < func_arg_count {
                                let temp_stack = return_values.len() - arg_count;
                                let continuation = LispFunc::create_continuation(
                                    f,
                                    func_arg_count,
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

                                update_stacks = Some((f, false));
                            } else {
                                update_stacks = Some((f, true));
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
            Instr::CheckType(arg_type) => {
                let same_type = arg_type == return_values.pop().unwrap().get_type();
                return_values.push(LispValue::Boolean(same_type));
            }
        }

        // This solution is not very elegant, but it's necessary
        // to please the borrowchecker in a safe manner.
        if let Some((next_func, push_stack)) = update_stacks {
            let next_arg_count = next_func.0.arg_count;
            let mut next_stack_ref =
                StackRef::new(next_func, return_values.len() - next_arg_count, state)?;
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
