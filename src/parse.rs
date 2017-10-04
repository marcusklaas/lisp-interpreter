use std::str::Chars;
use std::iter::Peekable;

use super::{BuiltIn, LispExpr, LispFunc, LispMacro, LispValue, State};

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

/// Token Iterator
struct Tokens<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'x> Tokens<'x> {
    fn from_str(literal: &str) -> Tokens {
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

            Token::OpVar(buf)
        }

        fn parse_integer(first: char, chars: &mut Peekable<Chars>) -> Token {
            let mut num = first.to_digit(10).unwrap();

            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                num = num * 10 + d;
                chars.next();
            }

            Token::Integer(u64::from(num))
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

pub fn parse_lisp_string(lit: &str, state: &mut State) -> Result<LispExpr, ParseError> {
    let mut tokens = Tokens::from_str(lit);
    let first_token = if let Some(t) = tokens.next() {
        t
    } else {
        return Err(ParseError::UnbalancedParens);
    };
    let res = parse_expr(first_token, &mut tokens, state)?;
    if !tokens.next().is_none() {
        return Err(ParseError::UnbalancedParens);
    }
    Ok(res)
}

// Tries to parse an iterator of tokens into a list of expressions.
// Expects the opening parenthesis to be stripped.
fn parse_call(tokens: &mut Tokens, state: &mut State) -> Result<Vec<LispExpr>, ParseError> {
    let mut stack = Vec::new();

    while let Some(token) = tokens.next() {
        if let Token::CloseParen = token {
            return Ok(stack);
        } else {
            let next_expr = parse_expr(token, tokens, state)?;
            stack.push(next_expr);
        }
    }

    Err(ParseError::UnbalancedParens)
}

fn parse_expr(
    token: Token,
    tokens: &mut Tokens,
    state: &mut State,
) -> Result<LispExpr, ParseError> {
    Ok(match token {
        Token::OpenParen => LispExpr::Call(parse_call(tokens, state)?),
        Token::CloseParen => return Err(ParseError::UnbalancedParens),
        Token::Integer(l) => LispExpr::Value(LispValue::Integer(l)),
        Token::OpVar(o) => if let Some(mac) = LispMacro::from_str(&o) {
            LispExpr::Macro(mac)
        } else if let Some(built_in) = BuiltIn::from_str(&o) {
            LispExpr::Value(LispValue::Function(LispFunc::BuiltIn(built_in)))
        } else if o == "#t" {
            LispExpr::Value(LispValue::Boolean(true))
        } else if o == "#f" {
            LispExpr::Value(LispValue::Boolean(false))
        } else {
            LispExpr::OpVar(state.intern(o))
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_double_parens() {
        let lit = "(())";
        let expected = Ok(LispExpr::Call(vec![LispExpr::Call(vec![])]));

        let result = parse_lisp_string(lit, &mut State::default());
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_simple_builtin() {
        let lit = "add1";
        let expected = Ok(LispExpr::Value(
            LispValue::Function(LispFunc::BuiltIn(BuiltIn::AddOne)),
        ));

        let result = parse_lisp_string(lit, &mut State::default());
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_double_closing_parens() {
        let lit = "(list))";
        let expected = Err(ParseError::UnbalancedParens);

        let result = parse_lisp_string(lit, &mut State::default());
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_integer() {
        let lit = "(55)";
        let expected = Ok(LispExpr::Call(
            vec![LispExpr::Value(LispValue::Integer(55))],
        ));

        let result = parse_lisp_string(lit, &mut State::default());
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_ok() {
        let lit = "(lambda (list 1 (cond 2 3) 9))";

        let expected = Ok(LispExpr::Call(vec![
            LispExpr::Macro(LispMacro::Lambda),
            LispExpr::Call(vec![
                LispExpr::Value(LispValue::Function(LispFunc::BuiltIn(BuiltIn::List))),
                LispExpr::Value(LispValue::Integer(1)),
                LispExpr::Call(vec![
                    LispExpr::Macro(LispMacro::Cond),
                    LispExpr::Value(LispValue::Integer(2)),
                    LispExpr::Value(LispValue::Integer(3)),
                ]),
                LispExpr::Value(LispValue::Integer(9)),
            ]),
        ]));

        let result = parse_lisp_string(lit, &mut State::default());
        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_unbalanced() {
        let lit = "(+ 1 (- 10 5)";
        let expected = Err(ParseError::UnbalancedParens);
        let result = parse_lisp_string(lit, &mut State::default());

        assert_eq!(expected, result);
    }

    #[test]
    fn parse_lisp_string_overbalanced() {
        let lit = "())";
        let expected = Err(ParseError::UnbalancedParens);
        let result = parse_lisp_string(lit, &mut State::default());

        assert_eq!(expected, result);
    }
}
