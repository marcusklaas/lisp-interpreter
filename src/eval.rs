use std::collections::HashMap;

use super::{LispValue, LispFunc, LispExpr, EvaluationError};

// So, at this point. Value bindings (or variables) are basically no
// different than function definitions without any additional arguments.
// We may want to get rid of the distinction at some point.

#[derive(Debug, Clone)]
pub struct State {
    bound: HashMap<String, LispValue>,
}

impl State {
    pub fn new() -> State {
        State {
            bound: [("#t", true), ("#f", false)]
                .into_iter()
                .map(|&(var_name, val)| {
                    (var_name.into(), NameBinding::Value(LispValue::Truth(val)))
                })
                .collect(),
        }
    }

    fn add_function(&mut self, name: String, args: Vec<String>, body: LispExpr) {
        let func = LispFunc { args, body };
        self.bound.insert(name, NameBinding::Function(func));
    }

    fn get_function(&self, name: &str) -> Option<&LispFunc> {
        match self.bound.get(name) {
            Some(&NameBinding::Function(ref x)) => Some(x),
            _ => None,
        }
    }

    fn get_variable_value(&self, var_name: &str) -> LispValue {
        match self.bound.get(var_name) {
            Some(ref val) => val.clone(),
            None => LispValue::BuiltIn(var_name.into()),
        }
    }

    pub fn set_variable(&mut self, var_name: &str, val: LispValue) {
        self.bound.insert(var_name.into(), NameBinding::Value(val));
    }
}

// TODO: should we simply add an `evauate` method to `LispExpr`?

pub fn evaluate_lisp_expr(
    expr: &LispExpr,
    state: &mut State,
) -> Result<LispValue, EvaluationError> {
    match *expr {
        LispExpr::Integer(n) => Ok(LispValue::Integer(n)),
        LispExpr::SubExpr(ref expr_vec) => {
            let inner_vec: Result<_, _> = expr_vec
                .iter()
                .map(|x| evaluate_lisp_expr(x, state))
                .collect();


            if let [LispValue::Function(f), ref tail..] = inner_vec[..] {
                // First OpVar could be a function, but also a variable.
                if let Some(res) = evaluate_lisp_fn(f, tail.iter(), state) {
                    return res;
                }
            }

            Ok(LispValue::SubValue(inner_vec?))
        }
        LispExpr::OpVar(ref x) => state.get_variable_value(x),
    }
}

fn save_args<'a, I>(args: I, expected_count: usize) -> Result<Vec<&'a LispExpr>, EvaluationError>
where
    I: Iterator<Item = &'a LispExpr>,
{
    let arg_vec = args.collect::<Vec<_>>();

    if arg_vec.len() == expected_count {
        Ok(arg_vec)
    } else {
        Err(EvaluationError::ArgumentCountMismatch)
    }
}

fn unitary_op<'a, I, F>(args: I, state: &mut State, f: F) -> Result<LispValue, EvaluationError>
where
    I: Iterator<Item = &'a LispExpr>,
    F: Fn(LispValue) -> Result<LispValue, EvaluationError>,
{
    save_args(args, 1).and_then(|arg_vec| evaluate_lisp_expr(arg_vec[0], state).and_then(f))
}

fn unitary_int_op<'a, I, F>(args: I, state: &mut State, f: F) -> Result<LispValue, EvaluationError>
where
    I: Iterator<Item = &'a LispExpr>,
    F: Fn(u64) -> Result<LispValue, EvaluationError>,
{
    unitary_op(args, state, |val| match val {
        LispValue::Integer(i) => f(i),
        _ => Err(EvaluationError::ArgumentTypeMismatch),
    })
}

fn unitary_list_op<'a, I, F>(args: I, state: &mut State, f: F) -> Result<LispValue, EvaluationError>
where
    I: Iterator<Item = &'a LispExpr>,
    F: Fn(Vec<LispValue>) -> Result<LispValue, EvaluationError>,
{
    unitary_op(args, state, |val| match val {
        LispValue::SubValue(vec) => f(vec),
        _ => Err(EvaluationError::ArgumentTypeMismatch),
    })
}

// Returns `None` when the function is not defined, `Some(Result<..>)` when it is.
fn evaluate_lisp_fn<'a, I>(
    f: LispFunc,
    args: I,
    state: &mut State,
) -> Option<Result<LispValue, EvaluationError>>
where
    I: Iterator<Item = &'a LispExpr>,
{
    match f {
        LispFunc::BuiltIn(ref fn_name) => Some(match fn_name {
            "null?" => unitary_list_op(args, state, |vec| Ok(LispValue::Truth(vec.is_empty()))),
            // So, usually cons prepends an element to a list, but since our internal
            // representation is a Vec, we'll actually append it. We may want to hide
            // this implementation detail by printing the elements in a list in the
            // reverse order in which they are stored.
            "cons" => {
                args.map(|arg| evaluate_lisp_expr(arg, state))
                    .collect::<Result<Vec<_>, _>>()
                    .and_then(|mut val_vec| match val_vec.pop().and_then(|list| {
                        val_vec.pop().map(|elt| (elt, list, val_vec.is_empty()))
                    }) {
                        Some((_, _, false)) |
                        None => Err(EvaluationError::ArgumentCountMismatch),
                        Some((x, LispValue::SubValue(mut vec), true)) => {
                            vec.push(x);
                            Ok(LispValue::SubValue(vec))
                        }
                        Some(..) => Err(EvaluationError::ArgumentTypeMismatch),
                    })
            }
            "cdr" => {
                unitary_list_op(args, state, |mut vec| match vec.pop() {
                    Some(_) => Ok(LispValue::SubValue(vec)),
                    None => Err(EvaluationError::EmptyList),
                })
            }
            // Congruent with the cons comment above, we'll actually have car return
            // the *last* element in our internal vector.
            "car" => {
                unitary_list_op(args, state, |mut vec| match vec.pop() {
                    Some(car) => Ok(car),
                    None => Err(EvaluationError::EmptyList),
                })
            }
            "cond" => {
                save_args(args, 3).and_then(|arg_vec| {
                    evaluate_lisp_expr(arg_vec[0], state).and_then(|cond_res| match cond_res {
                        LispValue::Truth(use_first) => {
                            let arg_index = if use_first { 1 } else { 2 };

                            evaluate_lisp_expr(arg_vec[arg_index], state)
                        }
                        _ => Err(EvaluationError::ArgumentTypeMismatch),
                    })
                })
            }
            "zero?" => unitary_int_op(args, state, |x| Ok(LispValue::Truth(x == 0))),
            "add1" => unitary_int_op(args, state, |x| Ok(LispValue::Integer(x + 1))),
            "sub1" => {
                unitary_int_op(args, state, |x| match x {
                    0 => Err(EvaluationError::SubZero),
                    i => Ok(LispValue::Integer(i - 1)),
                })
            }
            "+" => {
                save_args(args, 2).and_then(|arg_vec| {
                    evaluate_lisp_expr(arg_vec[0], state).and_then(|lhs| {
                        evaluate_lisp_expr(arg_vec[1], state).and_then(|rhs| match (lhs, rhs) {
                            (LispValue::Integer(x), LispValue::Integer(y)) => {
                                Ok(LispValue::Integer(x + y))
                            }
                            _ => Err(EvaluationError::ArgumentTypeMismatch),
                        })
                    })
                })
            }
            "define" => {
                save_args(args, 2).and_then(|arg_vec| {
                    evaluate_lisp_expr(arg_vec[1], state).and_then(|val| {
                        if let &LispExpr::OpVar(ref var_name) = arg_vec[0] {
                            state.set_variable(var_name, val.clone());
                            Ok(val)
                        } else {
                            // TODO: give this its own proper error
                            Err(EvaluationError::UndefinedFunction)
                        }
                    })
                })
            }
            "defun" => {
                save_args(args, 2).and_then(|arg_vec| {
                    match (arg_vec[0], arg_vec[1]) {
                        (&LispExpr::SubExpr(ref head_list), &LispExpr::SubExpr(_))
                            if !head_list.is_empty() => {
                            let names_vec: Option<Vec<String>> = head_list
                                .into_iter()
                                .map(|expr| match expr {
                                    &LispExpr::OpVar(ref name) => Some(name.clone()),
                                    _ => None,
                                })
                                .collect();

                            if let Some(mut names) = names_vec {
                                let func_name = names.remove(0);
                                state.add_function(func_name, names, arg_vec[1].clone());
                                // Return some dummy value.
                                Ok(LispValue::Integer(0))
                            } else {
                                Err(EvaluationError::MalformedDefinition)
                            }
                        }
                        (_, _) => Err(EvaluationError::MalformedDefinition),
                    }
                })
            }
            _ => Err(EvaluationError::UnknownVariable),
        }),
        Custom { args, body } => {
            // Function bodies now have access to the entire state,
            // which includes variables defined outside function scope.
            // should This is probably not something we want to allow.
            // We either create a clean state (with access to global
            // functions?) or check that this doesn't happen at function
            // definition.
            let mut new_state = state.clone();
            // step 1: evaluate all arguments to LispValues.
            let argument_res: Result<Vec<_>, _> = args.map(
                |x| evaluate_lisp_expr(x, &mut new_state),
            ).collect();

            argument_res.and_then(|argument_vec| {
                // step 2: check that number of variables matches.
                if argument_vec.len() != func.args.len() {
                    return Err(EvaluationError::ArgumentCountMismatch);
                }

                // step 3: map arguments to their names and add them to the State.
                for (arg_name, arg_value) in func.args.iter().zip(argument_vec.into_iter()) {
                    new_state.set_variable(arg_name, arg_value);
                }

                // step 4: evaluate function body.
                evaluate_lisp_expr(&func.body, &mut new_state)
            })
        }
    }
}
