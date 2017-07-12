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
        let lit = "(3 (+ 1 3) 0)";
        let expected = Ok(LispValue::SubValue(vec![
            LispValue::Integer(3),
            LispValue::Integer(4),
            LispValue::Integer(0),
        ]));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn run_nested_evaluation() {
        let lit = "(+ 10 (+ 1 10))";
        let expected = Ok(LispValue::Integer(21));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
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
        let mut state = State::new();
        state.set_variable("x", LispValue::Integer(5000));

        let lit = "(+ x x)";
        let expected = Ok(LispValue::Integer(10_000u64));
        let result: Result<_, LispError> =
            parse_lisp_string(lit).map_err(From::from).and_then(|ast| {
                evaluate_lisp_expr(&ast, &mut state).map_err(From::from)
            });

        assert_eq!(expected, result);
    }

    #[test]
    fn test_variable_setting() {
        let mut state = State::new();
        let lit = "(define x 5)";
        run_lisp_with_state(lit, &mut state).unwrap();

        let lit = "(+ x 7)";
        let expected = Ok(LispValue::Integer(12));
        let result = run_lisp_with_state(lit, &mut state);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_variable_list() {
        let mut state = State::new();
        state.set_variable("x", LispValue::Integer(3));

        let lit = "(x 1 (+ 1 x) 5)";
        let expected = Ok(LispValue::SubValue(vec![
            LispValue::Integer(3),
            LispValue::Integer(1),
            LispValue::Integer(4),
            LispValue::Integer(5),
        ]));
        let result = run_lisp_with_state(lit, &mut state);

        assert_eq!(expected, result);
    }

    #[test]
    fn eval_empty_list() {
        assert_eq!("()", run_lisp("()").unwrap().to_string());
    }
}
