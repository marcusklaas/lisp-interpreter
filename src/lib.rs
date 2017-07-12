#![feature(slice_patterns)]

use std::fmt;

pub mod parse;
pub mod eval;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LispExpr {
    Integer(u64),
    OpVar(String),
    SubExpr(Vec<LispExpr>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvaluationError {
    UndefinedFunction,
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
    SubValue(Vec<LispValue>),
}

impl fmt::Display for LispValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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

    fn run_lisp_with_state(lit: &str, state: &mut State) -> Result<LispValue, LispError> {
        Ok(evaluate_lisp_expr(&parse_lisp_string(lit)?, state)?)
    }

    fn run_lisp(lit: &str) -> Result<LispValue, LispError> {
        run_lisp_with_state(lit, &mut State::new())
    }

    fn check_lisp_ok<'i, I>(commands: I, expected_out: &str) where I: IntoIterator<Item = &'i str> {
        let mut state = State::new();
        let mut last_ret_val = None;

        for cmd in commands {
            let expr = parse_lisp_string(cmd).unwrap();
            last_ret_val = Some(evaluate_lisp_expr(&expr, &mut state).unwrap());
        }

        assert_eq!(expected_out, last_ret_val.unwrap().to_string());
    }

    // TODO: add tests for function definition and evaluation.

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
    fn run_simple_lisp_addition() {
        check_lisp_ok(vec!["(3 (+ 1 3) 0)"], "(3 4 0)");
    }

    #[test]
    fn run_nested_evaluation() {
        check_lisp_ok(vec!["(+ 10 (+ 1 10))"], "21");
    }

    #[test]
    fn too_few_arguments() {
        let lit = "(+ 10)";
        let expected = Err(LispError::Evaluation(
            EvaluationError::ArgumentCountMismatch,
        ));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn too_many_arguments() {
        let lit = "(+ 0 3 5)";
        let expected = Err(LispError::Evaluation(
            EvaluationError::ArgumentCountMismatch,
        ));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn unexpected_operator() {
        let lit = "(10 + 3)";
        let expected = Err(LispError::Evaluation(EvaluationError::UnknownVariable));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn undefined_function() {
        let lit = "(first (10 3))";
        let expected = Err(LispError::Evaluation(EvaluationError::UnknownVariable));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
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
}
