use std::str::Chars;
use std::iter::Peekable;
use std::collections::HashMap;

struct LispFunc {
    vars: Vec<String>,
    expr: LispExpr,
}

enum NameBinding {
    Value(LispValue),
    Function(LispFunc),
}

pub struct State {
    bound: HashMap<String, NameBinding>,
}

impl State {
    pub fn new() -> State {
        State { bound: HashMap::new() }
    }

    fn get_function(&self, name: &str) -> Option<&LispFunc> {
        match self.bound.get(name) {
            Some(&NameBinding::Function(ref x)) => Some(x),
            _ => None,
        }
    }

    fn get_variable_value(&self, var_name: &str) -> Result<LispValue, EvaluationError> {
        if let Some(&NameBinding::Value(ref val)) = self.bound.get(var_name) {
            Ok(val.clone())
        } else {
            Err(EvaluationError::UnknownVariable)
        }
    }

    fn set_variable(&mut self, var_name: &str, val: LispValue) {
        self.bound.insert(var_name.into(), NameBinding::Value(val));
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LispExpr {
    Integer(u64),
    OpVar(String),
    SubExpr(Vec<LispExpr>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    UnbalancedParens,
}

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Integer(u64),
    OpenParen,
    CloseParen,
    // Either an operator or a variable
    OpVar(String),
}

// Token Iterator.
struct Tokens<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'x> Tokens<'x> {
    fn from_str<'a>(literal: &'a str) -> Tokens<'a> {
        Tokens { chars: literal.chars().peekable() }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        fn is_function_char(c: &char) -> bool {
            match *c {
                '(' | ')' => false,
                x if x.is_whitespace() => false,
                _ => true,
            }
        }

        fn parse_func(first: char, chars: &mut Peekable<Chars>) -> Token {
            let mut buf = first.to_string();

            while let Some(true) = chars.peek().map(is_function_char) {
                buf.push(chars.next().unwrap());
            }

            Token::OpVar(buf)
        }

        fn parse_integer(first: char, chars: &mut Peekable<Chars>) -> Token {
            let mut num = first.to_digit(10).unwrap();

            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                num = num * 10 + d;
                chars.next();
            }

            Token::Integer(num as u64)
        }

        while let Some(c) = self.chars.next() {
            return match c {
                '(' => Some(Token::OpenParen),
                ')' => Some(Token::CloseParen),
                x if x.is_whitespace() => continue,
                x @ '0'...'9' => Some(parse_integer(x, &mut self.chars)),
                x => Some(parse_func(x, &mut self.chars)),
            };
        }

        None
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvaluationError {
    UndefinedFunction,
    UnexpectedOperator,
    ArgumentCountMismatch,
    ArgumentTypeMismatch,
    UnknownVariable,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LispValue {
    Integer(u64),
    SubValue(Vec<LispValue>),
}

pub fn evaluate_lisp_expr(
    expr: &LispExpr,
    state: &mut State,
) -> Result<LispValue, EvaluationError> {
    match *expr {
        LispExpr::Integer(n) => Ok(LispValue::Integer(n)),
        LispExpr::SubExpr(ref expr_vec) => {
            if let LispExpr::OpVar(ref name) = expr_vec[0] {
                // First OpVar could be a function, but also a variable.
                if let Some(res) = evaluate_lisp_fn(name, expr_vec[1..].iter(), state) {
                    return res;
                }
            }

            let inner_vec: Result<_, _> = expr_vec
                .iter()
                .map(|x| evaluate_lisp_expr(x, state))
                .collect();
            Ok(LispValue::SubValue(inner_vec?))
        }
        LispExpr::OpVar(ref x) => state.get_variable_value(x),
    }
}

fn get_binary_args<'a, I>(mut args: I) -> Result<(&'a LispExpr, &'a LispExpr), EvaluationError>
where
    I: Iterator<Item = &'a LispExpr>,
{
    match (args.next(), args.next(), args.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(EvaluationError::ArgumentCountMismatch),
    }
}

// Returns `None` when the function is not defined, `Some(Result<..>)` when it is.
fn evaluate_lisp_fn<'a, I>(
    fn_name: &str,
    mut args: I,
    state: &mut State,
) -> Option<Result<LispValue, EvaluationError>>
where
    I: Iterator<Item = &'a LispExpr>,
{
    Some(match fn_name {
        "+" => {
            get_binary_args(args).and_then(|(left, right)| {
                evaluate_lisp_expr(left, state).and_then(|lhs| {
                    evaluate_lisp_expr(right, state).and_then(|rhs| match (lhs, rhs) {
                        (LispValue::Integer(x), LispValue::Integer(y)) => {
                            Ok(LispValue::Integer(x + y))
                        }
                        _ => Err(EvaluationError::ArgumentTypeMismatch),
                    })
                })
            })
        }
        "define" => {
            get_binary_args(args).and_then(|(left, right)| {
                evaluate_lisp_expr(right, state).and_then(|val| {
                    if let &LispExpr::OpVar(ref var_name) = left {
                        state.set_variable(var_name, val.clone());
                        Ok(val)
                    } else {
                        // TODO: give this its own proper error
                        Err(EvaluationError::UndefinedFunction)
                    }
                })
            })
        }
        _ => return None,
    })
}

pub fn parse_lisp_string(lit: &str) -> Result<LispExpr, ParseError> {
    let mut tokens = Tokens::from_str(lit);
    // Strip the first token which we assume to be an opening paren, since
    // parse_lisp does not expect it.
    let _ = tokens.next();

    let result = parse_lisp(&mut tokens);

    match tokens.next() {
        None => result.map(|expr_vec| LispExpr::SubExpr(expr_vec)),
        Some(_) => Err(ParseError::UnbalancedParens),
    }
}

// Tries to parse an iterator of tokens into a list of expressions.
// Expects the opening parenthesis to be stripped.
fn parse_lisp(tokens: &mut Tokens) -> Result<Vec<LispExpr>, ParseError> {
    let mut stack = Vec::new();

    while let Some(token) = tokens.next() {
        let next_token = match token {
            Token::OpenParen => LispExpr::SubExpr(parse_lisp(tokens)?),
            Token::CloseParen => return Ok(stack),
            Token::Integer(l) => LispExpr::Integer(l),
            Token::OpVar(o) => LispExpr::OpVar(o),
        };
        stack.push(next_token);
    }

    Err(ParseError::UnbalancedParens)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn parse_double_parens() {
        let lit = "(())";
        let expected = Ok(LispExpr::SubExpr(vec![LispExpr::SubExpr(vec![])]));

        let result = parse_lisp_string(lit);
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_integer() {
        let lit = "(55)";
        let expected = Ok(LispExpr::SubExpr(vec![LispExpr::Integer(55)]));

        let result = parse_lisp_string(lit);
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_ok() {
        let lit = "(first (list 1 (+ 2 3) 9))";

        let expected = Ok(LispExpr::SubExpr(vec![
            LispExpr::OpVar("first".to_owned()),
            LispExpr::SubExpr(vec![
                LispExpr::OpVar("list".to_owned()),
                LispExpr::Integer(1),
                LispExpr::SubExpr(vec![
                    LispExpr::OpVar("+".to_owned()),
                    LispExpr::Integer(2),
                    LispExpr::Integer(3),
                ]),
                LispExpr::Integer(9),
            ]),
        ]));

        let result = parse_lisp_string(lit);
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_unbalanced() {
        let lit = "(+ 1 (- 10 5)";
        let expected = Err(ParseError::UnbalancedParens);
        let result = parse_lisp_string(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_overbalanced() {
        let lit = "())";
        let expected = Err(ParseError::UnbalancedParens);
        let result = parse_lisp_string(lit);

        assert_eq!(expected, result);
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
        let mut var_map = HashMap::new();
        var_map.insert("x".into(), NameBinding::Value(LispValue::Integer(5000)));
        let mut state = State { bound: var_map };

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
        run_lisp_with_state(lit, &mut state);

        let lit = "(+ x 7)";
        let expected = Ok(LispValue::Integer(12));
        let result = run_lisp_with_state(lit, &mut state);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_variable_list() {
        let mut var_map = HashMap::new();
        var_map.insert("x".into(), NameBinding::Value(LispValue::Integer(3)));
        let mut state = State { bound: var_map };

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
}
