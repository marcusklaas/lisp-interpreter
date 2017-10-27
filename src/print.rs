use std::iter::repeat;

use super::{CustomFunc, FinalizedExpr, LispFunc, LispValue, State};

pub fn print_value(val: &LispValue, state: &State, indent: usize) -> String {
    match *val {
        LispValue::Function(ref func) => print_lisp_func(func, state, indent),
        LispValue::Integer(i) => i.to_string(),
        LispValue::Boolean(true) => "#t".into(),
        LispValue::Boolean(false) => "#f".into(),
        LispValue::List(ref vec) => {
            let mut result = "(".to_string();

            for (idx, val) in vec.iter().enumerate() {
                if idx > 0 {
                    result.push(' ');
                }

                result.push_str(&print_value(val, state, indent));
            }

            result.push(')');
            result
        }
    }
}

fn indent_to_string(indent: usize) -> String {
    repeat(' ').take(indent * 4).collect()
}

fn format_list<'a, I: Iterator<Item = &'a FinalizedExpr>>(
    state: &State,
    indent: usize,
    first_item: &str,
    expr_list: I,
) -> String {
    let mut result = String::new();

    result.push('(');
    result.push_str(first_item);

    let new_indent = indent + 1;

    for (i, expr) in expr_list.into_iter().enumerate() {
        if i == 0 {
            result.push(' ');
        } else {
            result.push('\n');
            result.push_str(&indent_to_string(new_indent));
        };
        result.push_str(&print_finalized_expr(expr, state, new_indent));
    }

    result.push(')');
    result
}

fn print_custom_func(f: &CustomFunc, state: &State, indent: usize) -> String {
    let mut result = "(".to_owned();

    for i in 0..(f.0.arg_count) {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(&format!("${}", i));
    }

    result.push_str(&" -> ");
    result + &print_finalized_expr(&f.0.body, state, indent) + ")"
}

fn print_lisp_func(f: &LispFunc, state: &State, indent: usize) -> String {
    match *f {
        LispFunc::BuiltIn(name) => format!("{:?}", name),
        LispFunc::Custom(ref c) => print_custom_func(c, state, indent),
    }
}

fn print_finalized_expr(expr: &FinalizedExpr, state: &State, indent: usize) -> String {
    match *expr {
        FinalizedExpr::Argument(offset, scope, _move_status) => {
            format!("$[{}:{}]", scope, usize::from(offset))
        }
        FinalizedExpr::Value(ref v) => print_value(v, state, indent),
        FinalizedExpr::Variable(interned_name) => state.resolve_intern(interned_name).into(),
        FinalizedExpr::Cond(ref triple, ..) => {
            let (ref test_expr, ref true_expr, ref false_expr) = **triple;
            // FIXME: this is just too messy - how to better do this?
            let expr_iter = Some(&*test_expr)
                .into_iter()
                .chain(Some(&*true_expr).into_iter().chain(Some(&*false_expr)));
            format_list(state, indent, "cond", expr_iter)
        }
        FinalizedExpr::Lambda(arg_c, scope, ref body, _) => format!(
            "lambda ({}, {}) -> {}",
            arg_c,
            scope,
            print_finalized_expr(body, state, indent)
        ),
        FinalizedExpr::FunctionCall(ref funk, ref args, _is_tail_call, _is_self_call) => {
            format_list(
                state,
                indent,
                &print_finalized_expr(funk, state, indent),
                args.iter(),
            )
        }
    }
}
