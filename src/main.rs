use std::env;
use std::process::exit;
use std::str::Chars;
use std::iter::Peekable;

#[derive(Debug, PartialEq, Eq)]
enum LispExpr {
    Integer(u64),
    Operator(String),
    SubExpr(Vec<LispExpr>),
}

#[derive(Debug, PartialEq, Eq)]
enum ParseError {
    UnbalancedParens,
}

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Integer(u64),
    OpenParen,
    CloseParen,
    Operator(String),
}

// Token Iterator.
struct Tokens<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'x> Tokens<'x> {
    fn from_str<'a>(literal: &'a str) -> Tokens<'a> {
        Tokens {
            chars: literal.chars().peekable(),
        }
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

            Token::Operator(buf)
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


fn main() {
    let mut args = env::args();

    if args.len() != 2 {
        println!("Usage: lisp-parse <lisp string>");
        exit(1);
    }

    // Skip first argument as it's the program name.
    args.next();

    let lisp_literal = args.next().unwrap();
    let parse_result = parse_lisp_string(&lisp_literal);
    println!("Parse result: {:?}", parse_result);

    if let Ok(ref expr) = parse_result {
        let eval = evaluate_lisp_expr(expr);

        println!("Evaluation result: {:?}", eval);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum EvaluationError {
    UndefinedFunction,
    UnexpectedOperator,
    ArgumentCountMismatch,
    ArgumentTypeMismatch,
}

#[derive(Debug, PartialEq, Eq)]
enum LispValue {
    Integer(u64),
    SubValue(Vec<LispValue>),
}

fn evaluate_lisp_expr(expr: &LispExpr) -> Result<LispValue, EvaluationError> {
    match *expr {
        LispExpr::Integer(n) => Ok(LispValue::Integer(n)),
        LispExpr::Operator(_) => Err(EvaluationError::UnexpectedOperator),
        LispExpr::SubExpr(ref expr_vec) => {
            if let LispExpr::Operator(ref name) = expr_vec[0] {
                evaluate_lisp_fn(name, expr_vec[1..].iter().map(evaluate_lisp_expr))
            } else {
                let inner_vec: Result<_, _> = expr_vec.iter().map(evaluate_lisp_expr).collect();
                Ok(LispValue::SubValue(inner_vec?))
            }
        }
    }
}

fn evaluate_lisp_fn<I>(fn_name: &str, mut args: I) -> Result<LispValue, EvaluationError> 
    where I: Iterator<Item=Result<LispValue, EvaluationError>>
{
    match fn_name {
        "+" => {
            let first_arg = args.next();
            let second_arg = args.next();
            let rest = args.next();

            match (first_arg, second_arg, rest) {
                (Some(lhs), Some(rhs), None) => {
                    lhs.and_then(|left| rhs.and_then(|right| {
                        match (left, right) {
                            (LispValue::Integer(x), LispValue::Integer(y)) => Ok(LispValue::Integer(x + y)),
                            _ => Err(EvaluationError::ArgumentTypeMismatch),
                        }
                    }))
                }
                _ => Err(EvaluationError::ArgumentCountMismatch),
            }
        }
        _ => Err(EvaluationError::UndefinedFunction),
    }
}

fn parse_lisp_string(lit: &str) -> Result<LispExpr, ParseError> {
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
            Token::Operator(o) => LispExpr::Operator(o),
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

    fn run_lisp(lit: &str) -> Result<LispValue, LispError> {
        Ok(evaluate_lisp_expr(&parse_lisp_string(lit)?)?)
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
            LispExpr::Operator("first".to_owned()),
            LispExpr::SubExpr(vec![
                LispExpr::Operator("list".to_owned()),
                LispExpr::Integer(1),
                LispExpr::SubExpr(vec![
                    LispExpr::Operator("+".to_owned()),
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
            LispValue::Integer(0)
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
        let expected = Err(LispError::Evaluation(EvaluationError::ArgumentCountMismatch));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn too_many_arguments() {
        let lit = "(+ 0 3 5)";
        let expected = Err(LispError::Evaluation(EvaluationError::ArgumentCountMismatch));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn unexpected_operator() {
        let lit = "(10 + 3)";
        let expected = Err(LispError::Evaluation(EvaluationError::UnexpectedOperator));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn undefined_function() {
        let lit = "(first (10 3))";
        let expected = Err(LispError::Evaluation(EvaluationError::UndefinedFunction));
        let result = run_lisp(lit);

        assert_eq!(expected, result);
    }
}
