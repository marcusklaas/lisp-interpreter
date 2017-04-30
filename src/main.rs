use std::env;
use std::process::exit;
use std::str::Chars;
use std::iter::Peekable;

#[derive(Debug, PartialEq, Eq)]
enum LispLiteral {
    Integer(i64),
}

#[derive(Debug, PartialEq, Eq)]
enum LispExpr {
    Literal(LispLiteral),
    Operator(LispOperator),
    SubExpr(Vec<LispExpr>),
}

#[derive(Debug, PartialEq, Eq)]
enum ParseError {
    UnbalancedParens,
    UnexpectedChar(char),
}

#[derive(Debug, PartialEq, Eq)]
enum LispOperator {
    Add,
    Subtract,
    Func(String),
}

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Literal(LispLiteral),
    OpenParen,
    CloseParen,
    Operator(LispOperator),
}

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
    type Item = Result<Token, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        fn parse_func(first_char: char, chars: &mut Peekable<Chars>) -> Result<Token, ParseError> {
            let mut buf = first_char.to_string();

            while let Some(true) = chars.peek().map(|&c| is_func_char(c)) {
                buf.push(chars.next().unwrap());
            }

            Ok(Token::Operator(LispOperator::Func(buf)))
        }

        // FIXME: this should return a Result with Ok type LispLiteral.
        // Similar for parse_func.
        fn parse_integer(first_char: char, chars: &mut Peekable<Chars>) -> Result<Token, ParseError> {
            let mut num = first_char.to_digit(10).unwrap();

            while let Some(Some(x)) = chars.peek().map(|c| c.to_digit(10)) {
                num = num * 10 + x;
                chars.next();
            }

            Ok(Token::Literal(LispLiteral::Integer(num as i64)))
        }

        fn is_func_char(c: char) -> bool {
            match c {
                'a'...'z' | 'A'...'Z' | '_' => true,
                _ => false,
            }
        }

        // FIXME: we should be able to do this more elegantly
        loop {
            let next_char = self.chars.next().map(|c| {
                match c {
                    '(' => Some(Ok(Token::OpenParen)),
                    ')' => Some(Ok(Token::CloseParen)),
                    '+' => Some(Ok(Token::Operator(LispOperator::Add))),
                    '-' => Some(Ok(Token::Operator(LispOperator::Subtract))),
                    x@ '0'...'9' => Some(parse_integer(x, &mut self.chars)),
                    x if is_func_char(x) => Some(parse_func(x, &mut self.chars)),
                    ' ' => None,
                    x => Some(Err(ParseError::UnexpectedChar(x))),
                }
            });

            if let Some(None) = next_char {
                continue;
            } else {
                // TODO: remember to write a test case where parens are unbalanced!
                return next_char.map(Option::unwrap);
            }
        }
    }
}

fn main() {
    let mut args = env::args();

    if args.len() != 2 {
        println!("Useage: lisp-parse \"<lisp string>\"");
        exit(1);
    }

    // Skip first argument as it's the program name
    args.next();

    let lisp_literal = args.next().unwrap();
    println!("Lisp literal: \"{}\"", lisp_literal);

    let parse_result = parse_lisp_string(&lisp_literal);
    println!("Parse result: {:?}", parse_result);
}

fn parse_lisp_string(lit: &str) -> Result<Vec<LispExpr>, ParseError> {
    let mut tokens = Tokens::from_str(lit);
    // Strip the first token which we assume to be an opening paren, since
    // parse_lisp does not expect it
    let _ = tokens.next();

    let result = parse_lisp(&mut tokens);

    match tokens.next() {
        None => result,
        Some(_) => Err(ParseError::UnbalancedParens),
    }
}

// Tries to parse an iterator of tokens into a list of expressions.
// Expects the opening parenthesis to be stripped.
fn parse_lisp(tokens: &mut Tokens) -> Result<Vec<LispExpr>, ParseError> {
    let mut stack = Vec::new();

    while let Some(token) = tokens.next() {
        let next_token = match token? {
            Token::OpenParen => LispExpr::SubExpr(parse_lisp(tokens)?),
            Token::CloseParen => return Ok(stack),
            Token::Literal(l) => LispExpr::Literal(l),
            Token::Operator(o) => LispExpr::Operator(o),
        };
        stack.push(next_token);
    }

    Err(ParseError::UnbalancedParens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lisp_string_ok() {
        let lit = "(first (list 1 (+ 2 3) 9))";

        let expected = Ok(vec![
            LispExpr::Operator(LispOperator::Func("first".to_owned())),
            LispExpr::SubExpr(vec![
                LispExpr::Operator(LispOperator::Func("list".to_owned())),
                LispExpr::Literal(LispLiteral::Integer(1)),
                LispExpr::SubExpr(vec![
                    LispExpr::Operator(LispOperator::Add),
                    LispExpr::Literal(LispLiteral::Integer(2)),
                    LispExpr::Literal(LispLiteral::Integer(3)),
                ]),
                LispExpr::Literal(LispLiteral::Integer(9)),
            ]),
        ]);

        let result = super::parse_lisp_string(lit);
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_unbalanced() {
        let lit = "(+ 1 (- 10 5)";
        let expected = Err(ParseError::UnbalancedParens);
        let result = super::parse_lisp_string(lit);

        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_overbalanced() {
        let lit = "())";
        let expected = Err(ParseError::UnbalancedParens);
        let result = super::parse_lisp_string(lit);
        
        assert_eq!(expected, result);
    }
}
