use std::str::Chars;
use std::iter::Peekable;

use super::LispExpr;

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
}
