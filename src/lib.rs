#![feature(slice_patterns)]
#![feature(advanced_slice_patterns)]

pub mod parse;
pub mod eval;

use std::fmt;

#[derive(Debug, Clone)]
pub enum LispFunc {
    BuiltIn(String),
    Custom { args: Vec<String>, body: LispExpr },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LispExpr {
    Integer(u64),
    OpVar(String),
    SubExpr(Vec<LispExpr>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvaluationError {
    UnexpectedOperator,
    ArgumentCountMismatch,
    ArgumentTypeMismatch,
    SubZero,
    EmptyList,
    UnknownVariable,
    MalformedDefinition,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LispValue {
    Truth(bool),
    Integer(u64),
    Function(LispFunc),
    SubValue(Vec<LispValue>),
}

impl fmt::Display for LispValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &LispValue::Function(_) => write!(f, "func"),
            &LispValue::Integer(i) => write!(f, "{}", i),
            &LispValue::Truth(true) => write!(f, "#t"),
            &LispValue::Truth(false) => write!(f, "#f"),
            &LispValue::SubValue(ref vec) => {
                write!(f, "(")?;

                for (idx, val) in vec.iter().enumerate() {
                    if idx > 0 {
                        write!(f, " ")?;
                    }

                    write!(f, "{}", val)?;
                }

                write!(f, ")")
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::parse::{ParseError, parse_lisp_string};
    use super::eval::{State, evaluate_lisp_expr};
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
            last_ret_val = Some(evaluate_lisp_expr(&expr, &mut state)?);
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
                "(defun (add x y) (cond (zero? y) x (add (add1 x) (sub1 y))))",
                "(add 77 12)",
            ],
            "89",
        );
    }

    #[test]
    fn function_multiply() {
        check_lisp_ok(
            vec![
                "(defun (add x y) (cond (zero? y) x (add (add1 x) (sub1 y))))",
                "(defun (mult x y) (cond (zero? y) 0 (add x (mult x (sub1 y)))))",
                "(mult 7 3)",
            ],
            "21",
        );
    }

    #[test]
    fn function_def() {
        check_lisp_ok(vec!["(defun (add2 x) (add1 (add1 x)))", "(add2 5)"], "7");
    }

    #[test]
    fn is_null_empty_list() {
        check_lisp_ok(vec!["(null? ())"], "#t");
    }

    // #[test]
    // fn cdr() {
    //     check_lisp_ok(vec!["(cdr (1 2 3 4))"], "(2 3 4)");
    // }

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
            vec!["(zero? (0))"],
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
    fn run_simple_lisp_addition() {
        check_lisp_ok(vec!["(3 (+ 1 3) 0)"], "(3 4 0)");
    }

    #[test]
    fn run_nested_evaluation() {
        check_lisp_ok(vec!["(+ 10 (+ 1 10))"], "21");
    }

    #[test]
    fn too_few_arguments() {
        check_lisp_err(
            vec!["(+ 10)"],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[test]
    fn too_many_arguments() {
        check_lisp_err(
            vec!["(+ 0 3 5)"],
            LispError::Evaluation(EvaluationError::ArgumentCountMismatch),
        );
    }

    #[test]
    fn unexpected_operator() {
        check_lisp_err(
            vec!["(10 + 3)"],
            LispError::Evaluation(EvaluationError::UnknownVariable),
        );
    }

    #[test]
    fn undefined_function() {
        check_lisp_err(
            vec!["(first (10 3))"],
            LispError::Evaluation(EvaluationError::UnknownVariable),
        );
    }

    #[test]
    fn test_variable() {
        check_lisp_ok(vec!["(define x 5000)", "(+ x x)"], "10000");
    }

    #[test]
    fn test_variable_list() {
        check_lisp_ok(vec!["(define x 3)", "(x 1 (+ 1 x) 5)"], "(3 1 4 5)");
    }

    #[test]
    fn eval_empty_list() {
        check_lisp_ok(vec!["()"], "()");
    }

    #[test]
    fn map() {
        check_lisp_ok(
            vec![
                "(defun (map f xs) ((null? xs) xs (cons (f (cat xs)) (map f (cdr xs)))))",
                "(map add1 (1 2 3))",
            ],
            "(2 3 4)",
        );
    }
}
