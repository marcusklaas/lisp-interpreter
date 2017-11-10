use super::{builtin_instr, compile_finalized_expr, CustomFunc, EvaluationError, EvaluationResult,
            FinalizationContext, Instr, LispExpr, LispFunc, LispValue, StackOffset, State, TopExpr};
use std::iter;
use std::ops::Index;
use std::mem::{replace, transmute};
use std::default::Default;

fn unitary_list<F: Fn(&mut Vec<LispValue>) -> EvaluationResult<LispValue>>(
    stack: &mut Vec<LispValue>,
    f: F,
) -> EvaluationResult<()> {
    let reference = stack.last_mut().unwrap();

    *reference = if let LispValue::List(ref mut v) = *reference {
        f(v)?
    } else {
        return Err(EvaluationError::ArgumentTypeMismatch);
    };

    Ok(())
}

fn remove_old_arguments(stack: &mut Vec<LispValue>, start: StackOffset, end: StackOffset) {
    stack.splice(From::from(start)..From::from(end), iter::empty());
}

struct StackRef {
    instr_pointer: usize,
    #[allow(dead_code)] func: CustomFunc,
    stack_pointer: StackOffset,
    // This reference isn't really static - it refers to vector inside of
    // instr_vec. There's just no way to express this in Rust (I think!)
    instr_slice: &'static [Instr],
}

impl StackRef {
    fn new(
        func: CustomFunc,
        stack_pointer: StackOffset,
        state: &State,
    ) -> EvaluationResult<StackRef> {
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
    let (instructions, is_define) = match expr.into_top_expr()? {
        TopExpr::Define(name, sub_expr) => {
            let finalized_definition =
                sub_expr.finalize(&mut FinalizationContext::new(Some(name)))?;

            (
                compile_finalized_expr(finalized_definition.0, true, state)?,
                Some(name),
            )
        }
        TopExpr::Regular(sub_expr, _returns) => {
            (compile_finalized_expr(sub_expr, true, state)?, None)
        }
    };

    let result = run(instructions, state)?;

    if let Some(var_name) = is_define {
        state.set_variable(var_name, result, false)?;
        Ok(LispValue::List(Vec::new()))
    } else {
        Ok(result)
    }
}

fn run(instructions: Vec<Instr>, state: &State) -> EvaluationResult<LispValue> {
    let mut value_stack: Vec<LispValue> = Vec::new();
    let mut frame_stack = vec![];
    let mut frame = StackRef::new(
        CustomFunc::from_byte_code(0, instructions),
        StackOffset::default(),
        state,
    )?;

    'l: loop {
        frame.instr_pointer -= 1;

        match frame.instr_slice[frame.instr_pointer] {
            Instr::Return => {
                // Remove all values except for the last, which is the return value of
                // called function
                let top_index = StackOffset::from(value_stack.len() - 1);
                remove_old_arguments(&mut value_stack, frame.stack_pointer, top_index);

                if let Some(new_frame) = frame_stack.pop() {
                    frame = new_frame;
                } else {
                    break 'l;
                }
            }
            Instr::VarAddOne(offset) => if let LispValue::Integer(ref mut i) =
                *value_stack.get_mut(frame.stack_pointer + offset).unwrap()
            {
                *i += 1;
            } else {
                return Err(EvaluationError::ArgumentTypeMismatch);
            },
            Instr::CondZeroJumpDecr(offset, jump_size) => if let LispValue::Integer(ref mut i) =
                *value_stack.get_mut(frame.stack_pointer + offset).unwrap()
            {
                if *i == 0 {
                    frame.instr_pointer -= jump_size;
                } else {
                    *i -= 1;
                }
            } else {
                return Err(EvaluationError::ArgumentTypeMismatch);
            },
            Instr::VarCheckNull(offset) => {
                let head = if let LispValue::List(ref l) =
                    *value_stack.get(frame.stack_pointer + offset).unwrap()
                {
                    LispValue::Boolean(l.is_empty())
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                value_stack.push(head);
            }
            Instr::VarCheckZero(offset) => {
                let head = if let LispValue::Integer(i) =
                    *value_stack.get(frame.stack_pointer + offset).unwrap()
                {
                    LispValue::Boolean(i == 0)
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                value_stack.push(head);
            }
            Instr::VarSplit(offset) => {
                let head = if let LispValue::List(ref mut list) =
                    *value_stack.get_mut(frame.stack_pointer + offset).unwrap()
                {
                    if let Some(elem) = list.pop() {
                        elem
                    } else {
                        return Err(EvaluationError::EmptyList);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                value_stack.push(head);
            }
            Instr::VarReverseSplit(offset) => {
                // TODO: see if we can do this more efficiently/ elegantly
                let tail = {
                    let reference = value_stack.get_mut(frame.stack_pointer + offset).unwrap();
                    let mut head = if let LispValue::List(ref mut list) = *reference {
                        if let Some(elem) = list.pop() {
                            elem
                        } else {
                            return Err(EvaluationError::EmptyList);
                        }
                    } else {
                        return Err(EvaluationError::ArgumentTypeMismatch);
                    };

                    ::std::mem::swap(&mut head, reference);
                    head
                };

                value_stack.push(tail);
            }
            Instr::VarCar(offset) => {
                let head = if let LispValue::List(ref list) =
                    *value_stack.get(frame.stack_pointer + offset).unwrap()
                {
                    if let Some(elem) = list.last().cloned() {
                        elem
                    } else {
                        return Err(EvaluationError::EmptyList);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };

                value_stack.push(head);
            }
            Instr::Recurse(arg_count) => {
                if arg_count > 0 {
                    let top_index = frame.stack_pointer + StackOffset::from(frame.func.0.arg_count);
                    let bottom_index = top_index - StackOffset::from(arg_count);
                    remove_old_arguments(&mut value_stack, bottom_index, top_index);
                }
                frame.instr_pointer = frame.instr_slice.len();
            }
            Instr::CreateLambda(scope, arg_count, ref body, returns) => {
                // If there are any references to function arguments in
                // the lambda body, we should resolve them before
                // creating the lambda.
                // This enables us to do closures.
                let walked_body =
                    body.replace_args(scope, &mut value_stack[From::from(frame.stack_pointer)..]);
                let f = LispFunc::new_custom(arg_count, walked_body, returns);

                value_stack.push(LispValue::Function(f));
            }
            Instr::Jump(n) => {
                frame.instr_pointer -= n;
            }
            Instr::CondJump(n) => if let LispValue::Boolean(b) = value_stack.pop().unwrap() {
                if b {
                    frame.instr_pointer -= n;
                }
            } else {
                return Err(EvaluationError::ArgumentTypeMismatch);
            },
            Instr::PushValue(ref v) => {
                value_stack.push(v.clone());
            }
            Instr::CloneArgument(offset) => {
                let idx = frame.stack_pointer + offset;
                let value = (&value_stack[..]).index(idx).clone();
                value_stack.push(value);
            }
            Instr::MoveArgument(offset) => {
                let val = replace(
                    value_stack.get_mut(frame.stack_pointer + offset).unwrap(),
                    LispValue::Boolean(false),
                );
                value_stack.push(val);
            }
            // Pops a function off the value stack and applies it to the values
            // at the top of the value stack
            Instr::EvalFunction(arg_count, tail_call_args) => {
                let top_stack = value_stack.pop().unwrap();
                if let LispValue::Function(funk) = top_stack {
                    let (next_func, push_stack) = match funk {
                        LispFunc::BuiltIn(b) => {
                            // The performance of this solution is basically horrendous,
                            // but all the performant solutions are super messy.
                            // This shouldn't occur too often, though.
                            let func = CustomFunc::from_byte_code(
                                arg_count,
                                vec![Instr::Return, builtin_instr(b, arg_count)?],
                            );

                            (func, true)
                        }
                        LispFunc::Custom(f) => {
                            let func_arg_count = f.0.arg_count;

                            // Exactly right number of arguments. Let's evaluate.
                            if func_arg_count == arg_count {
                                if let Some(arg_reuse_count) = tail_call_args {
                                    if arg_reuse_count != func_arg_count {
                                        // Remove old arguments of the stack.
                                        let top_index = StackOffset::from(
                                            value_stack.len() - func_arg_count + arg_reuse_count,
                                        );
                                        let bottom_index = frame.stack_pointer
                                            + StackOffset::from(arg_reuse_count);
                                        remove_old_arguments(
                                            &mut value_stack,
                                            bottom_index,
                                            top_index,
                                        );
                                    }

                                    (f, false)
                                } else {
                                    // No need to add this frame to the frame stack when
                                    // we're just immediately going to return next
                                    (
                                        f,
                                        frame.instr_slice[frame.instr_pointer - 1] != Instr::Return,
                                    )
                                }
                            }
                            // Not enough arguments, let's create a lambda that takes
                            // the remainder.
                            else if arg_count < func_arg_count {
                                let temp_stack = value_stack.len() - arg_count;
                                let continuation = LispFunc::curry(
                                    f,
                                    func_arg_count,
                                    arg_count,
                                    value_stack.drain(temp_stack..),
                                );

                                value_stack.push(LispValue::Function(continuation));
                                continue;
                            }
                            // Too many arguments.
                            else {
                                return Err(EvaluationError::ArgumentCountMismatch);
                            }
                        }
                    };

                    // Create a new stack frame and replace the current one with it
                    let stack_pointer =
                        StackOffset::from(value_stack.len() - next_func.0.arg_count);
                    let next_frame = StackRef::new(next_func, stack_pointer, state)?;

                    // If the called function is not a tail call and there are instructions
                    // left in the calling function, push the old stack frame to the stack.
                    if push_stack {
                        frame_stack.push(replace(&mut frame, next_frame));
                    } else {
                        frame = next_frame;
                    }
                } else {
                    println!("Tried to apply {:?}", top_stack);
                    return Err(EvaluationError::NonFunctionApplication);
                }
            }
            Instr::List(arg_count) => {
                let len = value_stack.len();
                let new_vec = value_stack.split_off(len - arg_count);
                value_stack.push(LispValue::List(new_vec));
            }
            Instr::Car => unitary_list(&mut value_stack, |vec| match vec.pop() {
                Some(car) => Ok(car),
                None => Err(EvaluationError::EmptyList),
            })?,
            Instr::Cdr => {
                if let LispValue::List(ref mut v) = *value_stack.last_mut().unwrap() {
                    if v.pop().is_none() {
                        return Err(EvaluationError::EmptyList);
                    }
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };
            }
            Instr::CheckNull => unitary_list(
                &mut value_stack,
                |vec| Ok(LispValue::Boolean(vec.is_empty())),
            )?,
            Instr::AddOne => {
                if let LispValue::Integer(ref mut i) = *value_stack.last_mut().unwrap() {
                    *i += 1;
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::SubOne => {
                if let LispValue::Integer(ref mut i) = *value_stack.last_mut().unwrap() {
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
                let len = value_stack.len();
                let elt = value_stack.swap_remove(len - 2);

                if let LispValue::List(ref mut new_vec) = *value_stack.last_mut().unwrap() {
                    new_vec.push(elt);
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                }
            }
            Instr::CheckZero => {
                let reference = value_stack.last_mut().unwrap();
                let is_zero = if let LispValue::Integer(i) = *reference {
                    i == 0
                } else {
                    return Err(EvaluationError::ArgumentTypeMismatch);
                };
                *reference = LispValue::Boolean(is_zero);
            }
            Instr::CheckType(arg_type) => {
                let same_type = arg_type == value_stack.pop().unwrap().get_type();
                value_stack.push(LispValue::Boolean(same_type));
            }
        }
    }

    println!("{:?}", &value_stack);

    assert!(frame_stack.is_empty());
    assert_eq!(value_stack.len(), 1);
    Ok(value_stack.pop().unwrap())
}
