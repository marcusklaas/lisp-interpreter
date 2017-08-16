#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![feature(test, splice)]

extern crate test;

pub mod parse;
pub mod evaluator;

use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use evaluator::{State, Instr, compile_expr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomFunc {
    arg_count: usize,
    body: Rc<LispExpr>,
    byte_code: Rc<RefCell<Option<Vec<Instr>>>>,
}

impl CustomFunc {
    // FIXME: do a pass reducing the number of clones and stuff
    pub fn compile(&mut self, stack: &[LispValue], state: &State) -> Result<Vec<Instr>, EvaluationError> {
        {
            if let Some(ref vek) = *self.byte_code.borrow() {
                return Ok(vek.clone());
            } 
        }

        let (is_closure, new_bytes) = compile_expr((&*self.body).clone(), stack, state)?;
        if !is_closure {
            *(self.byte_code.borrow_mut()) = Some(new_bytes.clone());
        }
        
        Ok(new_bytes)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LispFunc {
    BuiltIn(&'static str),
    Custom(CustomFunc),
}

impl LispFunc {
    pub fn new_custom(args: Vec<String>, body: LispExpr, state: &State) -> LispFunc {
        LispFunc::Custom(CustomFunc {
            arg_count: args.len(),
            body: Rc::new(body.transform(&args[..], state, true)),
            byte_code: Rc::new(RefCell::new(None)),
        })
    }

    pub fn create_continuation(
        f: LispFunc,
        total_args: usize,
        supplied_args: usize,
        stack: &[LispValue],
    ) -> LispFunc {
        let arg_count = total_args - supplied_args;
        let mut call_vec = vec![LispExpr::Value(LispValue::Function(f))];
        call_vec.extend(stack[..supplied_args].iter().cloned().map(LispExpr::Value));
        call_vec.extend((0..total_args - supplied_args).map(LispExpr::Argument));

        LispFunc::Custom(CustomFunc {
            arg_count: arg_count,
            body: Rc::new(LispExpr::Call(call_vec, true)),
            byte_code: Rc::new(RefCell::new(None)),
        })
    }

    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispFunc::BuiltIn(name) => name.to_owned(),
            LispFunc::Custom(ref c) => c.pretty_print(indent),
        }
    }
}

fn indent_to_string(indent: usize) -> String {
    ::std::iter::repeat(' ').take(indent * 4).collect()
}

impl fmt::Display for LispFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pretty_print(0))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
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

// TODO: expressions with opvars / macros / arguments should probably have their
//       own type at some point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LispExpr {
    Macro(LispMacro),
    Value(LispValue),
    OpVar(String),
    // Offset from stack pointer on the return_values stack.
    Argument(usize),
    // Bool argument states whether the call is a
    // tail call.
    Call(Vec<LispExpr>, bool),
}

impl LispExpr {
    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispExpr::Argument(ref offset) => format!("${}", offset),
            LispExpr::Value(ref v) => v.pretty_print(indent),
            LispExpr::OpVar(ref name) => name.clone(),
            LispExpr::Macro(ref mac) => format!("{:?}", mac),
            LispExpr::Call(ref expr_vec, is_tail_call) => {
                let mut result = String::new();

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

    // Prepares a LispExpr for use in a lambda body, by mapping
    // variables to references argument indices and checking what
    // calls are tail calls.
    pub fn transform(self, args: &[String], state: &State, can_tail_call: bool) -> LispExpr {
        match self {
            x @ LispExpr::Value(_) => x,
            x @ LispExpr::Macro(_) => x,
            // This should not be possible. We shouldn't transform
            // an expression twice without resolving the arguments first.
            LispExpr::Argument(_) => unreachable!(),
            LispExpr::OpVar(name) => {
                // step 1: try to map it to an argument index
                if let Some(index) = args.into_iter().position(|a| a == &name) {
                    LispExpr::Argument(index)
                } else if let Some(v) = state.get_variable_value(&name) {
                    // step 2: if that fails, try to resolve it to a value in state
                    LispExpr::Value(v)
                } else {
                    LispExpr::OpVar(name)
                }
            }
            LispExpr::Call(vec, _) => {
                let do_tail_call = match (can_tail_call, vec.get(0)) {
                    // Special case for `cond`. Even though it is a function,
                    // its child expressions can still be tail calls.
                    (true, Some(&LispExpr::OpVar(ref name))) => name == "cond" && vec.len() == 4,
                    (
                        true,
                        Some(&LispExpr::Value(LispValue::Function(LispFunc::BuiltIn("cond")))),
                    ) => vec.len() == 4,
                    _ => false,
                };
                let tail_call_iter = (0..).map(|i| (i == 2 || i == 3) && do_tail_call);

                LispExpr::Call(
                    vec.into_iter()
                        .zip(tail_call_iter)
                        .map(|(e, can_tail)| e.transform(args, state, can_tail))
                        .collect(),
                    can_tail_call,
                )
            }
        }
    }

    // Resolves references to function arguments. Used when creating closures.
    pub fn replace_args(self, stack: &[LispValue]) -> (bool, LispExpr) {
        let mut is_closure = false;

        let next = match self {
            LispExpr::Argument(index) => {
                is_closure = true;
                LispExpr::Value(stack[index].clone())
            }
            LispExpr::Call(vec, is_tail_call) => LispExpr::Call(
                vec.into_iter().map(|e| {
                    let (replaced, exp) = e.replace_args(stack);
                    is_closure = is_closure || replaced;
                    exp
                }).collect(),
                is_tail_call,
            ),
            x => x,
        };

        (is_closure, next)
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
    TestOneTwoThree,
}

// TODO: add some convenience function for creating functions?
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LispValue {
    Boolean(bool),
    Integer(u64),
    Function(LispFunc),
    // TODO: this should be renamed to List
    SubValue(Vec<LispValue>),
}

impl LispValue {
    pub fn pretty_print(&self, indent: usize) -> String {
        match *self {
            LispValue::Function(ref func) => format!("[{}]", func.pretty_print(indent)),
            LispValue::Integer(i) => i.to_string(),
            LispValue::Boolean(true) => "#t".into(),
            LispValue::Boolean(false) => "#f".into(),
            LispValue::SubValue(ref vec) => {
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
            last_ret_val = Some(evaluator::eval(&expr, &mut state)?);
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
    fn transform_expr() {
        let expr = LispExpr::Call(
            vec![
                LispExpr::OpVar("x".into()),
                LispExpr::OpVar("#t".into()),
                LispExpr::Call(
                    vec![
                        LispExpr::Value(LispValue::Integer(5)),
                        LispExpr::OpVar("y".into()),
                    ],
                    false,
                ),
            ],
            false,
        );

        let transformed_expr = expr.transform(&["x".into(), "y".into()], &State::new(), true);

        let expected_transform = LispExpr::Call(
            vec![
                LispExpr::Argument(0),
                LispExpr::Value(LispValue::Boolean(true)),
                LispExpr::Call(
                    vec![
                        LispExpr::Value(LispValue::Integer(5)),
                        LispExpr::Argument(1),
                    ],
                    false,
                ),
            ],
            true,
        );

        assert_eq!(expected_transform, transformed_expr);
    }

    #[test]
    fn display_int_val() {
        let val = LispValue::Integer(5);
        assert_eq!("5", val.to_string());
    }

    #[test]
    fn display_list_val() {
        let val = LispValue::SubValue(vec![LispValue::Integer(1), LispValue::SubValue(vec![])]);
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
    fn function_def() {
        check_lisp_ok(
            vec!["(define add2 (lambda (x) (add1 (add1 x))))", "(add2 5)"],
            "7",
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
    fn unexpected_operator() {
        check_lisp_err(
            vec!["(10 + 3)"],
            LispError::Evaluation(EvaluationError::NonFunctionApplication),
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

    #[test]
    fn sort() {
        check_lisp_ok(
            vec![
                "(define filter (lambda (f xs) (cond (null? xs) (list) (cond (f (car xs)) (cons (car xs) (filter f (cdr xs))) (filter f (cdr xs))))))",
                "(define not (lambda (t) (cond t #f #t)))",
                "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
                "(define and (lambda (t1 t2) (cond t1 t2 #f)))",
                "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
                "(define sort (lambda (l) (cond (null? l) l (append (cons (car l) (sort (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort (filter (lambda (x) (> x (car l))) l))))))",
                "(sort (list 5 3 2 10 0 7))",
            ],
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
}
