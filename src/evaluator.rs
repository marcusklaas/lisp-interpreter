use super::{ArgType, BuiltIn, CustomFunc, EvaluationError, EvaluationResult, FinalizationContext,
            FinalizedExpr, InternedString, LispExpr, LispFunc, LispValue, MoveStatus, Scope,
            TopExpr};
// use super::specialization;
use std::collections::hash_map;
use std::collections::HashMap;
use std::iter;
use std::ops::Index;
use std::mem::{replace, transmute};
use std::default::Default;
use string_interner::StringInterner;

// TODO: ideally, this shouldn't be public - so
// we can guarantee that no invalid indices can be
// constructed
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct StateIndex(usize);

#[derive(Debug, Clone)]
pub struct State {
    interns: StringInterner<InternedString>,
    index_map: HashMap<InternedString, usize>,
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
            interns: StringInterner::new(),
            index_map: HashMap::new(),
            store: Vec::new(),
        }
    }
}

impl State {
    pub fn resolve_intern<'s>(&'s self, sym: InternedString) -> &'s str {
        // We trust that InternedString values have been created by us
        // and therefore must be valid symbols.
        unsafe { self.interns.resolve_unchecked(sym) }
    }

    pub fn intern<S: Into<String>>(&mut self, s: S) -> InternedString {
        self.interns.get_or_intern(s.into())
    }

    pub fn get_index(&self, var: InternedString) -> Option<StateIndex> {
        self.index_map.get(&var).map(|&i| StateIndex(i))
    }

    pub fn set_variable(
        &mut self,
        var_name: InternedString,
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
    PopAndSet(InternedString),
    /// Creates a custom function with given (scope level, argument count, function body) and pushes
    /// the result to the stack
    CreateLambda(Scope, usize, FinalizedExpr),

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

    /// Pushes the car of the variable with given offset to the stack.
    /// This is functionally equivalent to [CloneArgument(offset), Car]
    VarCar(usize),
    /// Checks whether a variable with given offset is zero and pushes
    /// the result to the stack
    VarCheckZero(usize),
    /// Checks whether a variable with given offset is an empty list and
    /// pushes the result to the stack
    VarCheckNull(usize),

    /// The most optimized instruction of all. Checks if the variable with
    /// given offset is zero. Jumps if it is, decrements it otherwise.
    /// Params mean (variable_offset, jump_size)
    CondZeroJumpDecr(usize, usize),
}

fn unitary_list<F: Fn(&mut Vec<LispValue>) -> EvaluationResult<LispValue>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> EvaluationResult<()> {
    let reference = stack.last_mut().unwrap();

    *reference = if let &mut LispValue::List(ref mut v) = reference {
        f(v)?
    } else {
        return Err(EvaluationError::ArgumentTypeMismatch);
    };

    Ok(())
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
                sub_expr.finalize(&mut FinalizationContext::new(Some(name)))?;

            let mut instructions = vec![Instr::PopAndSet(name)];
            instructions.extend(compile_finalized_expr(finalized_definition, state)?);
            Ok(instructions)
        }
        TopExpr::Regular(sub_expr) => compile_finalized_expr(sub_expr, state),
    }
}

// return the number of instructions written
fn inner_compile(
    expr: FinalizedExpr,
    state: &State,
    instructions: &mut Vec<Instr>,
) -> EvaluationResult<()> {
    match expr {
        FinalizedExpr::Argument(offset, _scope, MoveStatus::Unmoved) => {
            instructions.push(Instr::MoveArgument(offset));
        }
        FinalizedExpr::Argument(offset, _scope, _move_status) => {
            instructions.push(Instr::CloneArgument(offset));
        }
        FinalizedExpr::Value(v) => {
            instructions.push(Instr::PushValue(v));
        }
        FinalizedExpr::Variable(n) => if let Some(i) = state.get_index(n) {
            instructions.push(Instr::PushVariable(i));
        } else {
            return Err(EvaluationError::UnknownVariable(
                state.resolve_intern(n).into(),
            ));
        },
        FinalizedExpr::Cond(triple) => {
            let unpacked = *triple;
            let (test, true_expr, false_expr) = unpacked;
            let before_len = instructions.len();
            inner_compile(true_expr, state, instructions)?;
            let jump_size = instructions.len() - before_len;
            let before_len = instructions.len();
            instructions.push(Instr::Jump(jump_size));

            if let FinalizedExpr::FunctionCall(ref f_box, ref args, ..) = test {
                if let FinalizedExpr::Value(
                    LispValue::Function(LispFunc::BuiltIn(BuiltIn::CheckZero)),
                ) = **f_box
                {
                    if let Some(&FinalizedExpr::Argument(offset, scope, _)) = args.get(0) {
                        // OK, so at this point we know we are jumping conditionally
                        // on whether a function arg is zero.
                        // Next: make sure that every use of this argument in the false branch
                        // is within a sub1 call.
                        // If this is the case, replace all these sub1 calls by uses
                        // of the argument itself (maintaining its move status!).
                        // Then, encode the conditional jump, zero check and decrement using
                        // a single, superspecialized instruction.
                        if args.len() == 1 && false_expr.only_use_after_sub(offset, scope, false) {
                            let new_false_expr = false_expr.remove_subs_of(offset, scope);
                            inner_compile(new_false_expr, state, instructions)?;
                            let jump_size = instructions.len() - before_len;
                            instructions.push(Instr::CondZeroJumpDecr(offset, jump_size));

                            return Ok(());
                        }
                    }
                }
            }

            inner_compile(false_expr, state, instructions)?;
            let jump_size = instructions.len() - before_len;
            instructions.push(Instr::CondJump(jump_size));
            inner_compile(test, state, instructions)?;
        }
        FinalizedExpr::Lambda(arg_count, scope, body) => {
            instructions.push(Instr::CreateLambda(scope, arg_count, *body));
        }
        FinalizedExpr::FunctionCall(funk, args, is_tail_call, is_self_call) => {
            if let FinalizedExpr::Value(LispValue::Function(LispFunc::BuiltIn(bf))) = *funk {
                match bf {
                    BuiltIn::Car | BuiltIn::CheckNull | BuiltIn::CheckZero
                        if {
                            if let Some(&FinalizedExpr::Argument(_, _, _)) = args.get(0) {
                                args.len() == 1
                            } else {
                                false
                            }
                        } =>
                    {
                        // Inspection mode!
                        if let FinalizedExpr::Argument(offset, _scope, _move_status) = args[0] {
                            instructions.push(match bf {
                                BuiltIn::CheckNull => Instr::VarCheckNull(offset),
                                BuiltIn::CheckZero => Instr::VarCheckZero(offset),
                                BuiltIn::Car => Instr::VarCar(offset),
                                _ => unreachable!(),
                            });
                            return Ok(());
                        } else {
                            unreachable!()
                        }
                    }
                    f => instructions.push(builtin_instr(f, args.len())?),
                }
            } else if is_tail_call && is_self_call {
                instructions.push(Instr::Recurse(args.len()));
            } else {
                instructions.push(Instr::EvalFunction(args.len(), is_tail_call));
                inner_compile(*funk, state, instructions)?;
            };

            for expr in args.into_iter().rev() {
                inner_compile(expr, state, instructions)?;
            }
        }
    }

    Ok(())
}

pub fn compile_finalized_expr(expr: FinalizedExpr, state: &State) -> EvaluationResult<Vec<Instr>> {
    let mut instructions = Vec::with_capacity(32);

    inner_compile(expr, state, &mut instructions)?;

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
        if stack_ref.instr_pointer == 0 {
            let top_index = return_values.len() - 1;
            return_values.splice(stack_ref.stack_pointer..top_index, iter::empty());

            if let Some(new_stack_ref) = stax.pop() {
                stack_ref = new_stack_ref;
            } else {
                break 'l;
            }
        }

        stack_ref.instr_pointer -= 1;

        match stack_ref.instr_slice[stack_ref.instr_pointer] {
            Instr::CondZeroJumpDecr(offset, jump_size) => {
                if let &mut LispValue::Integer(ref mut i) = return_values
                    .get_mut(stack_ref.stack_pointer + offset)
                    .unwrap()
                {
                    if *i == 0 {
                        stack_ref.instr_pointer -= jump_size;
                    } else {
                        *i -= 1;
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::VarCheckNull(offset) => {
                let head = if let &LispValue::List(ref l) =
                    return_values.get(stack_ref.stack_pointer + offset).unwrap()
                {
                    LispValue::Boolean(l.is_empty())
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                return_values.push(head);
            }
            Instr::VarCheckZero(offset) => {
                let head = if let &LispValue::Integer(i) =
                    return_values.get(stack_ref.stack_pointer + offset).unwrap()
                {
                    LispValue::Boolean(i == 0)
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                return_values.push(head);
            }
            Instr::VarCar(offset) => {
                let head = if let &LispValue::List(ref list) =
                    return_values.get(stack_ref.stack_pointer + offset).unwrap()
                {
                    if let Some(elem) = list.last().cloned() {
                        elem
                    } else {
                        return Err(EvaluationError::EmptyList);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                return_values.push(head);
            }
            Instr::Recurse(arg_count) => {
                let top_index = return_values.len() - arg_count;
                return_values.splice(stack_ref.stack_pointer..top_index, iter::empty());
                stack_ref.instr_pointer = stack_ref.instr_slice.len();
            }
            Instr::CreateLambda(scope, arg_count, ref body) => {
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
                let val = replace(
                    return_values
                        .get_mut(stack_ref.stack_pointer + offset)
                        .unwrap(),
                    LispValue::Boolean(false),
                );
                return_values.push(val);
            }
            Instr::PushVariable(i) => {
                return_values.push(state[i].clone());
            }
            Instr::PopAndSet(var_name) => {
                state.set_variable(var_name, return_values.pop().unwrap(), false)?;
                return_values.push(LispValue::List(Vec::new()));
            }
            // Pops a function off the value stack and applies it
            Instr::EvalFunction(arg_count, is_tail_call) => {
                if let LispValue::Function(funk) = return_values.pop().unwrap() {
                    let (next_func, push_stack) = match funk {
                        LispFunc::BuiltIn(b) => {
                            // The performance of this solution is basically horrendous,
                            // but all the performant solutions are super messy.
                            // This shouldn't occur too often, though.
                            let func = CustomFunc::from_byte_code(
                                arg_count,
                                vec![builtin_instr(b, arg_count)?],
                            );

                            (func, true)
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
                                    return_values.drain(temp_stack..),
                                );

                                return_values.push(LispValue::Function(continuation));
                                continue;
                            }
                            // Exactly right number of arguments. Let's evaluate.
                            else if is_tail_call {
                                // Remove old arguments of the stack.
                                let top_index = return_values.len() - arg_count;
                                return_values
                                    .splice(stack_ref.stack_pointer..top_index, iter::empty());

                                (f, false)
                            } else {
                                (f, true)
                            }
                        }
                    };

                    let next_arg_count = next_func.0.arg_count;
                    let next_stack_ref =
                        StackRef::new(next_func, return_values.len() - next_arg_count, state)?;

                    if push_stack && stack_ref.instr_pointer != 0 {
                        stax.push(replace(&mut stack_ref, next_stack_ref));
                    } else {
                        stack_ref = next_stack_ref;
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
            Instr::Car => unitary_list(&mut return_values, |vec| match vec.pop() {
                Some(car) => Ok(car),
                None => Err(EvaluationError::EmptyList),
            })?,
            Instr::Cdr => {
                if let &mut LispValue::List(ref mut v) = return_values.last_mut().unwrap() {
                    if v.pop().is_none() {
                        return Err(EvaluationError::EmptyList);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };
            }
            Instr::CheckNull => unitary_list(&mut return_values, |vec| {
                Ok(LispValue::Boolean(vec.is_empty()))
            })?,
            Instr::AddOne => {
                if let &mut LispValue::Integer(ref mut i) = return_values.last_mut().unwrap() {
                    *i += 1;
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::SubOne => {
                if let &mut LispValue::Integer(ref mut i) = return_values.last_mut().unwrap() {
                    if *i > 0 {
                        *i -= 1;
                    } else {
                        return Err(EvaluationError::SubZero);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::Cons => {
                let len = return_values.len();
                let elt = return_values.swap_remove(len - 2);

                if let &mut LispValue::List(ref mut new_vec) = return_values.last_mut().unwrap() {
                    new_vec.push(elt);
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::CheckZero => {
                let reference = return_values.last_mut().unwrap();
                let is_zero = if let &mut LispValue::Integer(i) = reference {
                    i == 0
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };
                *reference = LispValue::Boolean(is_zero);
            }
            Instr::CheckType(arg_type) => {
                let same_type = arg_type == return_values.pop().unwrap().get_type();
                return_values.push(LispValue::Boolean(same_type));
            }
        }
    }

    assert!(stax.is_empty());
    assert_eq!(stack_ref.instr_pointer, 0);
    assert_eq!(return_values.len(), 1);
    Ok(return_values.pop().unwrap())
}
