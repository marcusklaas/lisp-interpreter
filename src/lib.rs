#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![feature(test, splice, slice_patterns)]

extern crate petgraph;
extern crate test;

pub mod parse;
#[macro_use]
pub mod evaluator;
mod specialization;

use std::fmt;
use std::iter::repeat;
use std::rc::Rc;
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use evaluator::{compile_expr, Instr, State, StateIndex};

type EvaluationResult<T> = Result<T, EvaluationError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomFunc {
    arg_count: usize,
    body: Rc<LispExpr>,
    byte_code: Rc<RefCell<Vec<Instr>>>,
}

impl Hash for CustomFunc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // TODO: check that this is right - are we not leaking memory?
        let ptr = Rc::into_raw(self.body.clone());
        state.write_usize(ptr as usize);
        unsafe {
            Rc::from_raw(ptr);
        }
    }
}

impl CustomFunc {
    // FIXME: do a pass reducing the number of clones and stuff
    // TODO: there's a lot of unnecessary checks with refcells
    // also, lot's of indirection. see if we can introduce a new type
    // that makes this more efficient
    pub fn compile(&self, state: &State) -> EvaluationResult<Rc<RefCell<Vec<Instr>>>> {
        {
            if !self.byte_code.borrow().is_empty() {
                return Ok(self.byte_code.clone());
            }
        }

        let mut body = (&*self.body).clone();
        let mut arg_vec: Vec<bool> = repeat(true).take(self.arg_count).collect();
        body.set_moves(&mut arg_vec);
        *(self.byte_code.borrow_mut()) = compile_expr(body, state)?;
        Ok(self.byte_code.clone())
    }

    pub fn pretty_print(&self, indent: usize) -> String {
        let mut result = String::new();

        for i in 0..self.arg_count {
            if i > 0 {
                result.push(' ');
            }
            result.push_str(&format!("${}", i));
        }

        result.push_str(&format!(" ->\n{}", indent_to_string(indent + 1)));
        result + &self.body.pretty_print(indent + 1)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum BuiltIn {
    AddOne,
    SubOne,
    Cons,
    Cdr,
    Car,
    List,
    CheckZero,
    CheckNull,
    CheckType(ArgType),
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum ArgType {
    Integer,
    Boolean,
    Function,
    List,
}

impl BuiltIn {
    fn from_str(s: &str) -> Option<BuiltIn> {
        match s {
            "add1" => Some(BuiltIn::AddOne),
            "sub1" => Some(BuiltIn::SubOne),
            "cons" => Some(BuiltIn::Cons),
            "cdr" => Some(BuiltIn::Cdr),
            "car" => Some(BuiltIn::Car),
            "list" => Some(BuiltIn::List),
            "zero?" => Some(BuiltIn::CheckZero),
            "null?" => Some(BuiltIn::CheckNull),
            "int?" => Some(BuiltIn::CheckType(ArgType::Integer)),
            "bool?" => Some(BuiltIn::CheckType(ArgType::Boolean)),
            "list?" => Some(BuiltIn::CheckType(ArgType::List)),
            "fun?" => Some(BuiltIn::CheckType(ArgType::Function)),
            _ => None,
        }
    }
}

impl fmt::Display for BuiltIn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: is there a way to ensure this is consistent with the
        // from_str function?
        let str = match *self {
            BuiltIn::AddOne => "add1",
            BuiltIn::SubOne => "sub1",
            BuiltIn::Cons => "cons",
            BuiltIn::Cdr => "cdr",
            BuiltIn::Car => "car",
            BuiltIn::List => "list",
            BuiltIn::CheckZero => "zero?",
            BuiltIn::CheckNull => "null?",
            BuiltIn::CheckType(ArgType::Function) => "fun?",
            BuiltIn::CheckType(ArgType::Boolean) => "bool?",
            BuiltIn::CheckType(ArgType::Integer) => "int?",
            BuiltIn::CheckType(ArgType::List) => "list?",
        };

        write!(f, "{}", str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LispFunc {
    BuiltIn(BuiltIn),
    Custom(CustomFunc),
}

impl LispFunc {
    pub fn new_custom(args: &[&str], body: LispExpr, state: &State) -> LispFunc {
        LispFunc::Custom(CustomFunc {
            arg_count: args.len(),
            body: Rc::new(body.transform(args, state, true)),
            byte_code: Rc::new(RefCell::new(Vec::new())),
        })
    }

    pub fn create_continuation(
        f: CustomFunc,
        total_args: usize,
        supplied_args: usize,
        stack: &[LispValue],
    ) -> LispFunc {
        let arg_count = total_args - supplied_args;
        let mut call_vec = vec![LispExpr::Value(LispValue::Function(LispFunc::Custom(f)))];
        call_vec.extend(stack[..supplied_args].iter().cloned().map(LispExpr::Value));
        call_vec.extend(
            (0..total_args - supplied_args).map(|o| LispExpr::Argument(o, true)),
        );

        LispFunc::Custom(CustomFunc {
            arg_count: arg_count,
            body: Rc::new(LispExpr::Call(call_vec, true, false)),
            byte_code: Rc::new(RefCell::new(Vec::new())),
        })
    }

    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispFunc::BuiltIn(name) => format!("{:?}", name),
            LispFunc::Custom(ref c) => c.pretty_print(indent),
        }
    }
}

fn indent_to_string(indent: usize) -> String {
    repeat(' ').take(indent * 4).collect()
}

impl fmt::Display for LispFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pretty_print(0))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LispMacro {
    Define,
    Cond,
    Lambda,
}

impl LispMacro {
    fn from_str(s: &str) -> Option<LispMacro> {
        match s {
            "define" => Some(LispMacro::Define),
            "cond" => Some(LispMacro::Cond),
            "lambda" => Some(LispMacro::Lambda),
            _ => None,
        }
    }
}

pub enum TopExpr {
    Define(String, FinalizedExpr),
    Regular(FinalizedExpr),
}

// TODO: replace bools by two variant enums
#[derive(Clone)]
pub enum FinalizedExpr {
    // Arg count, scope level, body
    Lambda(usize, usize, Box<FinalizedExpr>),
    // test expr, a branch, b branch
    // FIXME: don't do three boxes, but a single box containing triple
    Cond(Box<FinalizedExpr>, Box<FinalizedExpr>, Box<FinalizedExpr>),
    Variable(String),
    Value(LispValue),
    // Offset from stack pointer, scope level, moveable
    Argument(usize, usize, bool),
    // function, arguments, tail-call, self-call
    // FIXME: don't do double box
    FunctionCall(Box<FinalizedExpr>, Vec<FinalizedExpr>, bool, bool),
}

impl FinalizedExpr {
    // Resolves references to function arguments. Used when creating closures.
    pub fn replace_args(&self, scope_level: usize, stack: &[LispValue]) -> FinalizedExpr {
        match *self {
            FinalizedExpr::Argument(index, arg_scope, _is_move) if arg_scope == scope_level => {
                FinalizedExpr::Value(stack[index].clone())
            }
            FinalizedExpr::FunctionCall(ref head, ref vec, is_tail_call, is_self_call) => {
                FinalizedExpr::FunctionCall(
                    Box::new(head.replace_args(scope_level, stack)),
                    vec.iter()
                        .map(|e| e.replace_args(scope_level, stack))
                        .collect(),
                    is_tail_call,
                    is_self_call,
                )
            }
            FinalizedExpr::Cond(ref test, ref true_expr, ref false_expr) => FinalizedExpr::Cond(
                Box::new(test.replace_args(scope_level, stack)),
                Box::new(true_expr.replace_args(scope_level, stack)),
                Box::new(false_expr.replace_args(scope_level, stack)),
            ),
            FinalizedExpr::Lambda(arg_c, scope, ref body) => FinalizedExpr::Lambda(
                arg_c,
                scope,
                Box::new(body.replace_args(scope_level, stack)),
            ),
            ref x => x.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LispExpr {
    Macro(LispMacro),
    Value(LispValue),
    OpVar(String),
    // Offset from stack pointer on the return_values stack.
    // Bool indicates whether the value can be moved from arguments stack.
    Argument(usize, bool),
    // First bool argument states whether the call is a
    // tail call. Second states whether it is a self-call.
    Call(Vec<LispExpr>, bool, bool),
}

impl LispExpr {
    pub fn to_top_expr(self, state: &State) -> EvaluationResult<TopExpr> {
        if let LispExpr::Call(ref expr_list, _, _) = self {
            if let Some(&LispExpr::Macro(LispMacro::Define)) = expr_list.get(0) {
                return if let &[LispExpr::OpVar(ref n), ref definition] = &expr_list[1..] {
                    // FIXME: don't clone!
                    Ok(TopExpr::Define(
                        n.clone(),
                        definition
                            .clone()
                            .finalize(0, &HashMap::new(), state, true, Some(n))?,
                    ))
                } else {
                    Err(EvaluationError::BadDefine)
                };
            }
        }

        Ok(TopExpr::Regular(
            self.finalize(0, &HashMap::new(), state, true, None)?,
        ))
    }

    pub fn finalize(
        self,
        scope_level: usize,
        // maps symbols to (scope_level, offset, moveable)
        arguments: &HashMap<String, (usize, usize, Cell<bool>)>,
        state: &State,
        can_tail_call: bool,
        own_name: Option<&str>,
    ) -> EvaluationResult<FinalizedExpr> {
        Ok(match self {
            // TODO: LispExpr::Argument should probably be removed, right?
            // LispExpr::Argument(..) => unreachable!(),
            LispExpr::Argument(i, m) => FinalizedExpr::Argument(i, 0, m),
            LispExpr::Value(v) => FinalizedExpr::Value(v),
            LispExpr::OpVar(n) => {
                // So if we encounter a symbol, it could be two things:
                // a function argument, in which case it should be in the arguments map
                // a reference to something in our state.
                // Function arguments take precendence.
                if let Some(&(arg_scope, arg_offset, ref moveable)) = arguments.get(&n) {
                    let is_moveable = moveable.get();
                    moveable.replace(false);
                    FinalizedExpr::Argument(arg_offset, arg_scope, is_moveable)
                } else {
                    FinalizedExpr::Variable(n)
                }
            }
            LispExpr::Macro(..) => {
                return Err(EvaluationError::UnexpectedOperator);
            }
            LispExpr::Call(mut expr_list, _is_tail_call, _is_self_call) => {
                let head_expr = expr_list.remove(0);

                match head_expr {
                    LispExpr::Macro(LispMacro::Cond) => {
                        destructure!(expr_list, [test_expr, true_expr, false_expr], {
                            let false_expr_args = arguments.clone();
                            let finalized_false_expr = Box::new(false_expr.finalize(
                                scope_level,
                                &false_expr_args,
                                state,
                                true,
                                own_name,
                            )?);
                            let finalized_true_expr = Box::new(true_expr.finalize(
                                scope_level,
                                arguments,
                                state,
                                true,
                                own_name,
                            )?);

                            for key in arguments.keys() {
                                let new_value =
                                    arguments[key].2.get() && false_expr_args[key].2.get();
                                arguments[key].2.replace(new_value);
                            }

                            FinalizedExpr::Cond(
                                Box::new(test_expr.finalize(
                                    scope_level,
                                    arguments,
                                    state,
                                    true,
                                    own_name,
                                )?),
                                finalized_true_expr,
                                finalized_false_expr,
                            )
                        })
                    }
                    LispExpr::Macro(LispMacro::Lambda) => {
                        destructure!(expr_list, [arg_list, body], {
                            if let LispExpr::Call(ref arg_vec, _is_tail_call, _is_self_call) =
                                arg_list
                            {
                                // Add arguments to the arguments map, overwriting existing
                                // ones if they have the same symbol.
                                let mut new_arguments = arguments.clone();
                                let num_args = arg_vec.len();

                                for (offset, expr) in arg_vec.into_iter().enumerate() {
                                    let symbol = match *expr {
                                        LispExpr::OpVar(ref name) => Ok(&name[..]),
                                        _ => Err(EvaluationError::MalformedDefinition),
                                    }?;

                                    new_arguments.insert(
                                        symbol.to_owned(),
                                        (scope_level, offset, Cell::new(true)),
                                    );
                                }

                                FinalizedExpr::Lambda(
                                    num_args,
                                    scope_level,
                                    Box::new(body.finalize(
                                        scope_level + 1,
                                        &new_arguments,
                                        state,
                                        true,
                                        own_name,
                                    )?),
                                )
                            } else {
                                return Err(EvaluationError::ArgumentTypeMismatch);
                            }
                        })
                    }
                    // Defines should be caught by to_top_expr
                    LispExpr::Macro(LispMacro::Define) => unreachable!(),
                    // Function evaluation
                    _ => {
                        let is_self_call = if let LispExpr::OpVar(ref n) = head_expr {
                            if let Some(self_name) = own_name {
                                n == self_name
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        let finalized_args = expr_list
                            .into_iter()
                            .map(|e| {
                                e.finalize(scope_level, arguments, state, false, own_name)
                            })
                            .collect::<EvaluationResult<Vec<_>>>()?;
                        FinalizedExpr::FunctionCall(
                            Box::new(head_expr.finalize(
                                scope_level,
                                arguments,
                                state,
                                false,
                                own_name,
                            )?),
                            finalized_args,
                            can_tail_call,
                            is_self_call,
                        )
                    }
                }
            }
        })
    }

    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispExpr::Argument(ref offset, is_move) => {
                format!("{}${}", if is_move { "m" } else { "" }, offset)
            }
            LispExpr::Value(ref v) => v.pretty_print(indent),
            LispExpr::OpVar(ref name) => name.clone(),
            LispExpr::Macro(ref mac) => format!("{:?}", mac),
            LispExpr::Call(ref expr_vec, is_tail_call, is_self_call) => {
                let mut result = String::new();

                if is_self_call {
                    result.push('r');
                }
                if is_tail_call {
                    result.push('t');
                }

                result.push('{');

                for (idx, expr) in expr_vec.iter().enumerate() {
                    if idx > 0 {
                        result.push('\n');
                        result.push_str(&indent_to_string(indent));
                    }

                    result.push_str(&expr.pretty_print(indent));
                }

                result.push('}');
                result
            }
        }
    }

    // Flags which uses of function arguments can be compiled down
    // to moves.
    fn set_moves(&mut self, arg_movable: &mut [bool]) {
        match *self {
            LispExpr::Argument(i, ref mut is_move) => if arg_movable[i] {
                *is_move = true;
                arg_movable[i] = false;
            },
            LispExpr::Call(ref mut vec, _tail_call, _self_call) => {
                // Special case for `cond`. Each argument that hasn't been moved yet
                // can be moved in both branches.
                if let Some(&LispExpr::Macro(LispMacro::Cond)) = vec.get(0) {
                    if vec.len() == 4 {
                        // First check the branches, if an argument is moved in either of them,
                        // it cannot be used for the condition check.
                        let mut arg_movable_branch: Vec<bool> =
                            arg_movable.iter().cloned().collect();
                        vec[2].set_moves(&mut arg_movable_branch[..]);
                        vec[3].set_moves(arg_movable);

                        // For the condition check and above holds: an argument can only
                        // be moved when it was not moved in either branch.
                        for (a, b) in arg_movable.iter_mut().zip(arg_movable_branch.iter()) {
                            *a = *a && *b;
                        }
                        vec[1].set_moves(arg_movable);
                    }
                } else {
                    for e in vec.iter_mut().rev() {
                        e.set_moves(arg_movable)
                    }
                }
            }
            _ => {}
        }
    }

    // Prepares a LispExpr for use in a lambda body, by mapping
    // variables to references argument indices and checking what
    // calls are tail calls.
    pub fn transform(self, args: &[&str], state: &State, can_tail_call: bool) -> LispExpr {
        match self {
            x @ LispExpr::Value(_) => x,
            x @ LispExpr::Macro(_) => x,
            // This should not be possible. We shouldn't transform
            // an expression twice without resolving the arguments first.
            LispExpr::Argument(..) => unreachable!(),
            LispExpr::OpVar(name) => {
                if let Some(index) = args.into_iter().position(|a| a == &name) {
                    // step 1: try to map it to an argument index
                    LispExpr::Argument(index, false)
                } else if let Some(i) = state.get_index(&name) {
                    // step 2: if that fails, try to resolve it to a value in state
                    LispExpr::Value(state[i].clone())
                } else {
                    // else: leave it
                    LispExpr::OpVar(name)
                }
            }
            LispExpr::Call(vec, _tail_call, self_call) => {
                // Special case for `cond`. Even though it is a function,
                // its child expressions can still be tail calls.
                let do_tail_call = if let Some(&LispExpr::Macro(LispMacro::Cond)) = vec.get(0) {
                    can_tail_call && vec.len() == 4
                } else {
                    false
                };
                let tail_call_iter = (0..).map(|i| (i == 2 || i == 3) && do_tail_call);

                LispExpr::Call(
                    vec.into_iter()
                        .zip(tail_call_iter)
                        .map(|(e, can_tail)| e.transform(args, state, can_tail))
                        .collect(),
                    can_tail_call,
                    self_call,
                )
            }
        }
    }

    // Resolves references to function arguments. Used when creating closures.
    pub fn replace_args(&self, stack: &[LispValue]) -> LispExpr {
        match *self {
            LispExpr::Argument(index, _is_move) => LispExpr::Value(stack[index].clone()),
            LispExpr::Call(ref vec, is_tail_call, is_self_call) => LispExpr::Call(
                vec.iter().map(|e| e.replace_args(stack)).collect(),
                is_tail_call,
                is_self_call,
            ),
            ref x => x.clone(),
        }
    }

    pub fn flag_self_calls(&mut self, name: &str) {
        if let LispExpr::Call(ref mut vec, _tail_call, ref mut is_self_call) = *self {
            if let Some(&LispExpr::OpVar(ref n)) = vec.get(0) {
                *is_self_call = n == name;
            }

            for e in vec {
                e.flag_self_calls(name);
            }
        }
    }
}

impl fmt::Display for LispExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pretty_print(0))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvaluationError {
    UnexpectedOperator,
    ArgumentCountMismatch,
    ArgumentTypeMismatch,
    EmptyListEvaluation,
    NonFunctionApplication,
    SubZero,
    EmptyList,
    UnknownVariable(String),
    MalformedDefinition,
    BadDefine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LispValue {
    Boolean(bool),
    Integer(u64),
    Function(LispFunc),
    List(Vec<LispValue>),
}

impl LispValue {
    pub fn get_type(&self) -> ArgType {
        match *self {
            LispValue::Boolean(..) => ArgType::Boolean,
            LispValue::Integer(..) => ArgType::Integer,
            LispValue::Function(..) => ArgType::Function,
            LispValue::List(..) => ArgType::List,
        }
    }

    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispValue::Function(ref func) => format!("[{}]", func.pretty_print(indent)),
            LispValue::Integer(i) => i.to_string(),
            LispValue::Boolean(true) => "#t".into(),
            LispValue::Boolean(false) => "#f".into(),
            LispValue::List(ref vec) => {
                let mut result = "(".to_string();

                for (idx, val) in vec.iter().enumerate() {
                    if idx > 0 {
                        result.push(' ');
                    }

                    result.push_str(&val.pretty_print(indent));
                }

                result.push(')');
                result
            }
        }
    }
}

impl fmt::Display for LispValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pretty_print(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::parse::{parse_lisp_string, ParseError};
    use super::evaluator::State;
    use std::convert::From;

    #[derive(Debug, PartialEq, Eq)]
    enum LispError {
        Parse(ParseError),
        Evaluation(EvaluationError),
    }

    impl From<EvaluationError> for LispError {
        fn from(err: EvaluationError) -> LispError {
            LispError::Evaluation(err)
        }
    }

    impl From<ParseError> for LispError {
        fn from(err: ParseError) -> LispError {
            LispError::Parse(err)
        }
    }

    fn check_lisp<'i, I>(commands: I) -> Result<LispValue, LispError>
    where
        I: IntoIterator<Item = &'i str>,
    {
        let mut state = State::new();
        let mut last_ret_val = None;

        for cmd in commands {
            let expr = parse_lisp_string(cmd)?;
            last_ret_val = Some(evaluator::eval(expr, &mut state)?);
        }

        Ok(last_ret_val.unwrap())
    }

    fn check_lisp_ok<'i, I>(commands: I, expected_out: &str)
    where
        I: IntoIterator<Item = &'i str>,
    {
        assert_eq!(expected_out, check_lisp(commands).unwrap().to_string());
    }

    fn check_lisp_err<'i, I>(commands: I, expected_err: LispError)
    where
        I: IntoIterator<Item = &'i str>,
    {
        assert_eq!(expected_err, check_lisp(commands).unwrap_err());
    }

    #[test]
    fn add_bytecode() {
        let add = check_lisp(vec![
            "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
            "(add 0 0)",
            "(car (list add))",
        ]).unwrap();

        match add {
            LispValue::Function(LispFunc::Custom(f)) => {
                assert_eq!(
                    vec![
                        Instr::MoveArgument(0),
                        Instr::Jump(1),
                        Instr::Recurse(2),
                        Instr::SubOne,
                        Instr::MoveArgument(1),
                        Instr::AddOne,
                        Instr::MoveArgument(0),
                        Instr::CondJump(6),
                        Instr::CheckZero,
                        Instr::CloneArgument(1),
                    ],
                    f.byte_code.borrow().clone()
                );
            }
            _ => panic!("expected function!"),
        }
    }

    #[test]
    fn transform_expr() {
        let expr = LispExpr::Call(
            vec![
                LispExpr::OpVar("x".into()),
                LispExpr::Value(LispValue::Boolean(true)),
                LispExpr::Call(
                    vec![
                        LispExpr::Value(LispValue::Integer(5)),
                        LispExpr::OpVar("y".into()),
                    ],
                    false,
                    false,
                ),
            ],
            false,
            false,
        );

        let transformed_expr = expr.transform(&["x".into(), "y".into()], &State::new(), true);

        let expected_transform = LispExpr::Call(
            vec![
                LispExpr::Argument(0, false),
                LispExpr::Value(LispValue::Boolean(true)),
                LispExpr::Call(
                    vec![
                        LispExpr::Value(LispValue::Integer(5)),
                        LispExpr::Argument(1, false),
                    ],
                    false,
                    false,
                ),
            ],
            true,
            false,
        );

        assert_eq!(expected_transform, transformed_expr);
    }

    #[test]
    fn shadowing() {
        check_lisp_ok(
            vec![
                "(define f (lambda (x) (lambda (x) x)))",
                "(list ((f #t) 3) ((f (list)) #f))",
            ],
            "(3 #f)",
        );
    }

    #[test]
    fn display_int_val() {
        let val = LispValue::Integer(5);
        assert_eq!("5", val.to_string());
    }

    #[test]
    fn display_list_val() {
        let val = LispValue::List(vec![LispValue::Integer(1), LispValue::List(vec![])]);
        assert_eq!("(1 ())", val.to_string());
    }

    #[test]
    fn function_add() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(add 77 12)",
            ],
            "89",
        );
    }

    #[test]
    fn function_multiply() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define mult (lambda (x y) (cond (zero? y) 0 (add x (mult x (sub1 y))))))",
                "(mult 7 3)",
            ],
            "21",
        );
    }

    #[test]
    fn map2_zip() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define or (lambda (x y) (cond x #t y)))",
                "(define zip (lambda (x y) (cond (or (null? x) (null? y)) (list) (cons (list (car x) (car y)) (zip (cdr x) (cdr y))))))",
                "(define map2 (lambda (f l) (cond (null? l) (list) (cons (f (car (cdr (car l))) (car (car l))) (map2 f (cdr l))))))",
                "(map2 add (zip (list 1 2 3 4 5) (list 0 20 40 60)))",
            ],
            "(2 23 44 65)",
        );
    }

    #[test]
    fn function_def() {
        check_lisp_ok(
            vec!["(define add2 (lambda (x) (add1 (add1 x))))", "(add2 5)"],
            "7",
        );
    }

    #[test]
    fn variable_overwrite() {
        check_lisp_ok(vec!["(define x 1)", "(define x 1000)", "(add1 x)"], "1001");
    }

    #[test]
    fn check_int() {
        check_lisp_ok(
            vec![
                "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
                "(map int? (list 1 2 #t (list 3 4) add1 0))",
            ],
            "(#t #t #f #f #f #t)",
        );
    }

    #[test]
    fn check_bool() {
        check_lisp_ok(
            vec![
                "(list (bool? #t) (bool? #f) (bool? 5) (bool? (list #t)) (bool? bool?))",
            ],
            "(#t #t #f #f #f)",
        );
    }

    #[test]
    fn check_fun() {
        check_lisp_ok(
            vec![
                "(list (fun? #t) (fun? fun?) (fun? 0) (fun? bool?) (fun? (lambda (x) #t)) (fun? #t))",
            ],
            "(#f #t #f #t #t #f)",
        );
    }

    #[test]
    fn check_list() {
        check_lisp_ok(
            vec!["(list (list? list) (list? list?) (list? (list)))"],
            "(#f #f #t)",
        );
    }

    #[test]
    fn is_null_empty_list() {
        check_lisp_ok(vec!["(null? (list))"], "#t");
    }

    #[test]
    fn cdr() {
        check_lisp_ok(vec!["(cdr (list 1 2 3 4))"], "(1 2 3)");
    }

    #[test]
    fn is_zero_of_zero() {
        check_lisp_ok(vec!["(zero? 0)"], "#t");
    }

    #[test]
    fn is_zero_of_nonzero() {
        check_lisp_ok(vec!["(zero? 5)"], "#f");
    }

    #[test]
    fn is_zero_of_list() {
        check_lisp_err(
            vec!["(zero? (list 0))"],
            LispError::Evaluation(EvaluationError::ArgumentTypeMismatch),
        );
    }

    #[test]
    fn is_zero_two_args() {
        check_lisp_err(
            vec!["(zero? 0 0)"],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[test]
    fn too_few_arguments() {
        check_lisp_err(
            vec!["(add1)"],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[test]
    fn too_many_arguments() {
        check_lisp_err(
            vec!["(lambda f (x) (add1 x) ())"],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[test]
    fn non_function_app() {
        check_lisp_err(
            vec!["(10 3)"],
            LispError::Evaluation(EvaluationError::NonFunctionApplication),
        );
    }

    #[test]
    fn unexpected_operator() {
        check_lisp_err(
            vec!["(cond cond cond cond)"],
            LispError::Evaluation(EvaluationError::UnexpectedOperator),
        );
    }

    #[test]
    fn undefined_function() {
        check_lisp_err(
            vec!["(first (list 10 3))"],
            LispError::Evaluation(EvaluationError::UnknownVariable("first".into())),
        );
    }

    #[test]
    fn test_variable_list() {
        check_lisp_ok(
            vec![
                "(define x 3)",
                "(define + (lambda (x y) (cond (zero? y) x (+ (add1 x) (sub1 y)))))",
                "(list x 1 (+ 1 x) 5)",
            ],
            "(3 1 4 5)",
        );
    }

    #[test]
    fn eval_empty_list() {
        check_lisp_ok(vec!["(list)"], "()");
    }

    #[test]
    fn map() {
        check_lisp_ok(
            vec![
                "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
                "(map add1 (list 1 2 3))",
            ],
            "(2 3 4)",
        );
    }

    #[test]
    fn lambda() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define mult (lambda (x y) (cond (zero? y) 0 (add x (mult x (sub1 y))))))",
                "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
                "(map (lambda (x) (mult x x)) (list 1 2 3))",
            ],
            "(1 4 9)",
        );
    }

    const SORT_COMMANDS: &[&str] = &[
        "(define filter (lambda (f xs) (cond (null? xs) (list) (cond (f (car xs)) (cons (car xs) (filter f (cdr xs))) (filter f (cdr xs))))))",
        "(define not (lambda (t) (cond t #f #t)))",
        "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
        "(define and (lambda (t1 t2) (cond t1 t2 #f)))",
        "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
        "(define sort (lambda (l) (cond (null? l) l (append (cons (car l) (sort (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort (filter (lambda (x) (> x (car l))) l))))))",
    ];

    #[test]
    fn sort() {
        check_lisp_ok(
            SORT_COMMANDS
                .into_iter()
                .cloned()
                .chain(Some("(sort (list 5 3 2 10 0 7))").into_iter()),
            "(0 2 3 5 7 10)",
        );
    }

    #[test]
    fn closures() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
                "(map (lambda (f) (f 10)) (map (lambda (n) (lambda (x) (add x n))) (list 1 2 3 4 5 6 7 8 9 10)))",
            ],
            "(11 12 13 14 15 16 17 18 19 20)",
        );
    }

    #[test]
    fn list_closure() {
        assert!(check_lisp(vec!["(list add1 ((lambda (f x) (f x)) sub1))"]).is_ok());
    }

    #[test]
    fn curry() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define sum3 (lambda (x y z) (add x (add y z))))",
                "(define sum2and5 (sum3 5))",
                "(sum2and5 10 20)",
            ],
            "35",
        );
    }

    #[test]
    fn cyclic_func_calls() {
        check_lisp_ok(
            vec![
                "(define <' (lambda (x y) (cond (zero? y) #f (< x (sub1 y)))))",
                "(define < (lambda (x y) (cond (zero? x) (cond (zero? y) #f #t) (<' (sub1 x) y))))",
                "(list (< 1 2) (< 0 1) (< 2 2) (< 1 1) (< 1 0) (< 2 1))",
            ],
            "(#t #t #f #f #f #f)",
        );
    }

    #[test]
    fn range() {
        check_lisp_ok(
            vec![
                "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
                "(define range (lambda (start end) (cond (> end start) (cons end (range start (sub1 end))) (list start))))",
                "(range 1 5)",
            ],
            "(1 2 3 4 5)",
        );
    }

    #[test]
    fn zero_arg_function_call() {
        check_lisp_err(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(add)",
            ],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[bench]
    fn bench_add(b: &mut super::test::Bencher) {
        b.iter(|| {
            check_lisp(vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(add 100 100)",
            ])
        });
    }

    #[bench]
    fn bench_mult(b: &mut super::test::Bencher) {
        b.iter(|| {
            check_lisp(vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define mult (lambda (x y) (cond (zero? y) 0 (add (mult x (sub1 y)) x))))",
                "(mult 10 100)",
            ])
        });
    }

    #[bench]
    fn bench_sort(b: &mut super::test::Bencher) {
        let mut state = State::new();

        for cmd in SORT_COMMANDS {
            let expr = parse_lisp_string(cmd).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| {
            let expr = parse_lisp_string("(sort (list 5 1 0 3 2 10 30 0 7 1))").unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        });
    }
}
