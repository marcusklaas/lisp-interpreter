#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(test, feature(test))]
#![feature(splice, slice_patterns, slice_get_slice, collections_range)]

extern crate string_interner;
#[cfg(test)]
extern crate test;

pub mod parse;
pub mod evaluator;

use std::mem::{replace, transmute_copy};
use std::convert::From;
use std::fmt;
use std::iter::repeat;
use std::rc::Rc;
use std::cell::UnsafeCell;
use std::collections::hash_map;
use std::collections::HashMap;
use string_interner::StringInterner;
use std::slice::SliceIndex;
use std::ops::{Add, Index, IndexMut, Sub};

macro_rules! destructure {
    ( $y:ident, $x:expr ) => {{$x}};

    ( $iter:ident, [ $( $i:ident ),* ], $body:expr ) => {
        {
            if let ($( Some( $i), )* None) = {
                ( $( destructure!($i, $iter.next()), )* $iter.next() )
            } {
                Ok($body)
            } else {
                Err(EvaluationError::ArgumentCountMismatch)
            }?
        }
    };
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct StackOffset(u32);

impl StackOffset {
    fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<StackOffset> for usize {
    fn from(t: StackOffset) -> Self {
        t.0 as usize
    }
}

impl From<usize> for StackOffset {
    fn from(t: usize) -> Self {
        StackOffset(t as u32)
    }
}

impl Default for StackOffset {
    fn default() -> Self {
        StackOffset(0)
    }
}

impl Add for StackOffset {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        StackOffset(self.0 + other.0)
    }
}

impl Sub for StackOffset {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        StackOffset(self.0 - other.0)
    }
}

impl SliceIndex<[LispValue]> for StackOffset {
    type Output = LispValue;

    fn get(self, slice: &[LispValue]) -> Option<&Self::Output> {
        slice.get(self.to_usize())
    }

    fn get_mut(self, slice: &mut [LispValue]) -> Option<&mut Self::Output> {
        slice.get_mut(self.to_usize())
    }

    unsafe fn get_unchecked(self, _slice: &[LispValue]) -> &Self::Output {
        unimplemented!()
    }

    unsafe fn get_unchecked_mut(self, _slice: &mut [LispValue]) -> &mut Self::Output {
        unimplemented!()
    }

    fn index(self, slice: &[LispValue]) -> &Self::Output {
        slice.index(self.to_usize())
    }

    fn index_mut(self, slice: &mut [LispValue]) -> &mut Self::Output {
        slice.index_mut(self.to_usize())
    }
}

type EvaluationResult<T> = Result<T, EvaluationError>;

#[derive(Debug)]
struct InnerCustomFunc {
    arg_count: usize,
    body: FinalizedExpr,
    byte_code: UnsafeCell<Vec<Instr>>,
}

#[derive(Debug, Clone)]
pub struct State {
    interns: StringInterner<InternedString>,
    store: HashMap<InternedString, LispValue>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            interns: StringInterner::new(),
            store: HashMap::new(),
        }
    }
}

impl State {
    fn resolve_intern(&self, sym: InternedString) -> &str {
        // We trust that InternedString values have been created by us
        // and therefore must be valid symbols.
        unsafe { self.interns.resolve_unchecked(sym) }
    }

    pub fn intern<S: Into<String>>(&mut self, s: S) -> InternedString {
        self.interns.get_or_intern(s.into())
    }

    fn get(&self, var: InternedString) -> Option<&LispValue> {
        self.store.get(&var)
    }

    pub fn set_variable(
        &mut self,
        var_name: InternedString,
        val: LispValue,
        allow_override: bool,
    ) -> EvaluationResult<()> {
        match self.store.entry(var_name) {
            hash_map::Entry::Occupied(..) if !allow_override => Err(EvaluationError::BadDefine),
            hash_map::Entry::Occupied(mut occ_entry) => {
                *occ_entry.get_mut() = val;
                Ok(())
            }
            hash_map::Entry::Vacant(vac_entry) => {
                vac_entry.insert(val);
                Ok(())
            }
        }
    }

    pub fn get_variable_keys(&self) -> Vec<String> {
        self.store
            .keys()
            .map(|&intern| self.resolve_intern(intern).into())
            .collect()
    }
}

/// A custom function is the product of a lambda call. It is basically just
/// another expression that gets evaluated when the function is called.
/// We keep track of the number of arguments so that we know when enough
/// of them have been supplied to evaluate.
/// This function will not be compiled until it is called. This allows us
/// to refer to definitions that may not have existed at the time of
/// construction. Otherwise we wouldn't be able to create mutually recursive
/// functions.
/// To ensure that every function is only compiled once - an expensive
/// operation - all details of a function are kept in a reference counted
/// structure. This makes copying functions cheap.
#[derive(Debug, Clone)]
pub struct CustomFunc(Rc<InnerCustomFunc>);

impl PartialEq for CustomFunc {
    fn eq(&self, other: &CustomFunc) -> bool {
        unsafe { transmute_copy::<_, usize>(&self.0) == transmute_copy(&other.0) }
    }
}

impl Eq for CustomFunc {}

impl CustomFunc {
    fn compile<'s>(&'s self, state: &State) -> EvaluationResult<&'s [Instr]> {
        unsafe {
            let borrowed = &*self.0.byte_code.get();
            if !borrowed.is_empty() {
                Ok(&borrowed[..])
            } else {
                let mut_borrowed = &mut *self.0.byte_code.get();
                *mut_borrowed = compile_finalized_expr(self.0.body.clone(), state)?;
                mut_borrowed.insert(0, Instr::Return);
                Ok(&mut_borrowed[..])
            }
        }
    }

    fn from_byte_code(arg_count: usize, bytecode: Vec<Instr>) -> Self {
        CustomFunc(Rc::new(InnerCustomFunc {
            arg_count: arg_count,
            // dummy value
            body: FinalizedExpr::Value(LispValue::Boolean(false)),
            byte_code: UnsafeCell::new(bytecode),
        }))
    }

    pub fn pretty_print(&self, state: &State, indent: usize) -> String {
        let mut result = String::new();

        for i in 0..self.0.arg_count {
            if i > 0 {
                result.push(' ');
            }
            result.push_str(&format!("${}", i));
        }

        result.push_str(&format!(" ->\n{}", indent_to_string(indent + 1)));
        result + &self.0.body.pretty_print(state, indent + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Scope(u32);

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Scope {
    fn default() -> Self {
        Scope(0)
    }
}

impl Scope {
    fn next(self) -> Self {
        Scope(self.0 + 1)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum BuiltIn {
    AddOne,
    SubOne,
    Cons,
    Cdr,
    Car,
    List,
    CheckZero,
    CheckNull,
    CheckType(ArgType),
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ArgType {
    Integer,
    Boolean,
    Function,
    List,
}

impl BuiltIn {
    fn from_str(s: &str) -> Option<BuiltIn> {
        match s {
            "add1" => Some(BuiltIn::AddOne),
            "sub1" => Some(BuiltIn::SubOne),
            "cons" => Some(BuiltIn::Cons),
            "cdr" => Some(BuiltIn::Cdr),
            "car" => Some(BuiltIn::Car),
            "list" => Some(BuiltIn::List),
            "zero?" => Some(BuiltIn::CheckZero),
            "null?" => Some(BuiltIn::CheckNull),
            "int?" => Some(BuiltIn::CheckType(ArgType::Integer)),
            "bool?" => Some(BuiltIn::CheckType(ArgType::Boolean)),
            "list?" => Some(BuiltIn::CheckType(ArgType::List)),
            "fun?" => Some(BuiltIn::CheckType(ArgType::Function)),
            _ => None,
        }
    }
}

impl fmt::Display for BuiltIn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: is there a way to ensure this is consistent with the
        // from_str function?
        let str = match *self {
            BuiltIn::AddOne => "add1",
            BuiltIn::SubOne => "sub1",
            BuiltIn::Cons => "cons",
            BuiltIn::Cdr => "cdr",
            BuiltIn::Car => "car",
            BuiltIn::List => "list",
            BuiltIn::CheckZero => "zero?",
            BuiltIn::CheckNull => "null?",
            BuiltIn::CheckType(ArgType::Function) => "fun?",
            BuiltIn::CheckType(ArgType::Boolean) => "bool?",
            BuiltIn::CheckType(ArgType::Integer) => "int?",
            BuiltIn::CheckType(ArgType::List) => "list?",
        };

        write!(f, "{}", str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LispFunc {
    BuiltIn(BuiltIn),
    Custom(CustomFunc),
}

impl LispFunc {
    fn new_custom(arg_count: usize, body: FinalizedExpr) -> LispFunc {
        LispFunc::Custom(CustomFunc(Rc::new(InnerCustomFunc {
            arg_count: arg_count,
            body: body,
            byte_code: UnsafeCell::new(Vec::new()),
        })))
    }

    fn curry<I: Iterator<Item = LispValue>>(
        f: CustomFunc,
        total_args: usize,
        supplied_args: usize,
        args: I,
    ) -> LispFunc {
        let arg_count = total_args - supplied_args;
        let funk = Box::new(FinalizedExpr::Value(
            LispValue::Function(LispFunc::Custom(f)),
        ));
        let arg_vec = args.map(FinalizedExpr::Value)
            .chain((0..total_args - supplied_args).map(From::from).map(|o| {
                FinalizedExpr::Argument(o, Scope::default(), VariableConstraint::Unconstrained)
            }))
            .collect();

        Self::new_custom(
            arg_count,
            FinalizedExpr::FunctionCall(funk, arg_vec, true, false),
        )
    }

    pub fn pretty_print(&self, state: &State, indent: usize) -> String {
        match *self {
            LispFunc::BuiltIn(name) => format!("{:?}", name),
            LispFunc::Custom(ref c) => c.pretty_print(state, indent),
        }
    }
}

fn indent_to_string(indent: usize) -> String {
    repeat(' ').take(indent * 4).collect()
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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

/// In our implementation, defines can only happen at the top level of
/// an expression. To enforce in the types, `FinalizedExpr` does not contain
/// a Define variant.
#[derive(Debug, Clone)]
enum TopExpr {
    Define(InternedString, LispExpr),
    Regular(FinalizedExpr),
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// TODO: explain how `FinalizedExpression` is different from `LispExpr`
enum FinalizedExpr {
    // Arg count, scope level, body
    Lambda(usize, Scope, Box<FinalizedExpr>),
    // test expr, true branch, false branch.
    // false branch returns, true branch returns, tail call status of cond
    // TODO: clean up this variant.
    Cond(
        Box<(FinalizedExpr, FinalizedExpr, FinalizedExpr)>,
        bool,
        bool,
        TailCallStatus,
    ),
    Variable(InternedString),
    Value(LispValue),
    Argument(StackOffset, Scope, VariableConstraint),
    // function, arguments, tail-call, self-call
    FunctionCall(Box<FinalizedExpr>, Vec<FinalizedExpr>, bool, bool),
}

impl FinalizedExpr {
    /// Replaces subexpressions of form sub1(arg) by arg, where arg is the function
    /// argument with given offset and scope.
    fn remove_subs_of(self, offset: StackOffset, scope: Scope) -> Self {
        match self {
            FinalizedExpr::FunctionCall(f, args, tail_call, self_call) => {
                if let FinalizedExpr::Value(
                    LispValue::Function(LispFunc::BuiltIn(BuiltIn::SubOne)),
                ) = *f
                {
                    if let Some(&FinalizedExpr::Argument(e_offset, e_scope, _)) = args.get(0) {
                        if args.len() == 1 && offset == e_offset && e_scope == scope {
                            return args.into_iter().next().unwrap();
                        }
                    }
                }

                FinalizedExpr::FunctionCall(
                    Box::new(f.remove_subs_of(offset, scope)),
                    args.into_iter()
                        .map(|a| a.remove_subs_of(offset, scope))
                        .collect(),
                    tail_call,
                    self_call,
                )
            }
            FinalizedExpr::Cond(boks, false_expr_returns, true_expr_returns, tail_call_status) => {
                let triple = *boks;
                let (test, true_expr, false_expr) = triple;

                FinalizedExpr::Cond(
                    Box::new((
                        test.remove_subs_of(offset, scope),
                        true_expr.remove_subs_of(offset, scope),
                        false_expr.remove_subs_of(offset, scope),
                    )),
                    false_expr_returns,
                    true_expr_returns,
                    tail_call_status,
                )
            }
            FinalizedExpr::Lambda(a, b, body) => {
                FinalizedExpr::Lambda(a, b, Box::new(body.remove_subs_of(offset, scope)))
            }
            x => x,
        }
    }

    /// Checks whether this expression only uses the variable at the given
    /// offset as an argument to the sub1 function.
    fn only_use_after_sub(&self, offset: StackOffset, scope: Scope, parent_sub: bool) -> bool {
        match *self {
            FinalizedExpr::Argument(e_offset, e_scope, _move) => {
                e_offset != offset || e_scope != scope || parent_sub
            }
            FinalizedExpr::Cond(ref boks, ..) => {
                let (ref test, ref true_expr, ref false_expr) = **boks;
                test.only_use_after_sub(offset, scope, false)
                    && true_expr.only_use_after_sub(offset, scope, false)
                    && false_expr.only_use_after_sub(offset, scope, false)
            }
            FinalizedExpr::Variable(..) | FinalizedExpr::Value(..) => true,
            FinalizedExpr::Lambda(_, _, ref body) => body.only_use_after_sub(offset, scope, false),
            FinalizedExpr::FunctionCall(ref f, ref args, _, _) => {
                let is_sub = FinalizedExpr::Value(
                    LispValue::Function(LispFunc::BuiltIn(BuiltIn::SubOne)),
                ) == **f;

                f.only_use_after_sub(offset, scope, is_sub)
                    && args.iter()
                        .all(|a| a.only_use_after_sub(offset, scope, is_sub))
            }
        }
    }

    // Resolves references to function arguments. Used when creating closures.
    fn replace_args(&self, scope_level: Scope, stack: &mut [LispValue]) -> FinalizedExpr {
        match *self {
            FinalizedExpr::Argument(index, arg_scope, move_status) if arg_scope < scope_level => {
                if move_status == VariableConstraint::Unconstrained {
                    FinalizedExpr::Value(replace(&mut stack[index], LispValue::Boolean(false)))
                } else {
                    FinalizedExpr::Value(stack[index].clone())
                }
            }
            FinalizedExpr::FunctionCall(ref head, ref vec, is_tail_call, is_self_call) => {
                FinalizedExpr::FunctionCall(
                    Box::new(head.replace_args(scope_level, stack)),
                    vec.iter()
                        .map(|e| e.replace_args(scope_level, stack))
                        .collect(),
                    is_tail_call,
                    is_self_call,
                )
            }
            FinalizedExpr::Cond(
                ref triple,
                false_expr_returns,
                true_expr_returns,
                tail_call_status,
            ) => {
                let (ref test, ref true_expr, ref false_expr) = **triple;
                FinalizedExpr::Cond(
                    Box::new((
                        test.replace_args(scope_level, stack),
                        true_expr.replace_args(scope_level, stack),
                        false_expr.replace_args(scope_level, stack),
                    )),
                    false_expr_returns,
                    true_expr_returns,
                    tail_call_status,
                )
            }
            FinalizedExpr::Lambda(arg_c, scope, ref body) => FinalizedExpr::Lambda(
                arg_c,
                scope,
                Box::new(body.replace_args(scope_level, stack)),
            ),
            ref x => x.clone(),
        }
    }

    pub fn pretty_print(&self, state: &State, indent: usize) -> String {
        match *self {
            FinalizedExpr::Argument(offset, scope, move_status) => format!(
                "{}$({:?}, {})",
                if move_status == VariableConstraint::Unconstrained {
                    "m"
                } else {
                    ""
                },
                offset,
                scope
            ),
            FinalizedExpr::Value(ref v) => v.pretty_print(state, indent),
            FinalizedExpr::Variable(interned_name) => state.resolve_intern(interned_name).into(),
            FinalizedExpr::Cond(ref triple, ..) => {
                let (ref test, ref true_expr, ref false_expr) = **triple;
                let expr_iter = Some(&*true_expr).into_iter().chain(Some(&*false_expr));
                format_list(
                    state,
                    indent,
                    "cond".to_owned(),
                    &test.pretty_print(state, indent),
                    expr_iter,
                )
            }
            FinalizedExpr::Lambda(arg_c, scope, ref body) => format!(
                "lambda ({}, {}) -> {}",
                arg_c,
                scope,
                body.pretty_print(state, indent)
            ),
            FinalizedExpr::FunctionCall(ref funk, ref args, is_tail_call, is_self_call) => {
                let prefix = match (is_self_call, is_tail_call) {
                    (true, true) => "r".to_owned(),
                    (false, true) => "t".to_owned(),
                    (_, _) => String::new(),
                };

                format_list(
                    state,
                    indent,
                    prefix,
                    &funk.pretty_print(state, indent),
                    args.iter(),
                )
            }
        }
    }
}

/// Used in finalization process
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum TailCallStatus {
    CanTailCall,
    CannotTailCall,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum Instr {
    /// Calls the function we're currently in with the given number of arguments
    /// at the top of the stack
    Recurse(usize),
    /// Evaluates the function at the top of the stack with given number of arguments.
    /// The second parameter indicates whether this is a tail call, and if so, whether
    /// we can skip reuse the arguments
    EvalFunction(usize, Option<usize>),
    /// Pops a value from the stack and adds it to the state at the given name
    PopAndSet(InternedString),
    /// Creates a custom function with given (scope level, argument count, function body) and pushes
    /// the result to the stack
    CreateLambda(Scope, usize, Box<FinalizedExpr>),
    /// Pops the stack reference and removes everything from the stack pointer
    /// upwards from the value stack except for the top value
    Return,

    /// Skips the given number of instructions
    Jump(usize),
    /// Pops boolean value from stack and conditionally jumps a number of instructions
    CondJump(usize),
    /// Pushes a value to the stack
    PushValue(LispValue),
    /// Clones the n'th argument to the function and pushes it to the stack
    CloneArgument(StackOffset),
    /// Moves the n'th argument to the function to the top of the stack and replaces
    /// it with a dummy value.
    MoveArgument(StackOffset),

    // Built-in instructions
    AddOne,
    SubOne,
    Cons,
    Cdr,
    Car,
    List(usize),
    CheckZero,
    CheckNull,
    CheckType(ArgType),

    /// Pushes the car of the variable with given offset to the stack.
    /// This is functionally equivalent to [CloneArgument(offset), Car]
    VarCar(StackOffset),
    /// Identical to VarCar, except that it replaces the variable at the
    /// given offset by its Cdr.
    VarSplit(StackOffset),
    /// Replaces the list at the given offset by its head and places the
    /// tail at the top of the value stack.
    VarReverseSplit(StackOffset),
    /// Checks whether a variable with given offset is zero and pushes
    /// the result to the stack
    VarCheckZero(StackOffset),
    /// Checks whether a variable with given offset is an empty list and
    /// pushes the result to the stack
    VarCheckNull(StackOffset),
    /// Increments variable at given offset
    VarAddOne(StackOffset),

    /// The most optimized instruction of all. Checks if the variable with
    /// given offset is zero. Jumps if it is, decrements it otherwise.
    /// Params mean (variable_offset, jump_size)
    CondZeroJumpDecr(StackOffset, usize),
}

fn format_list<'a, I: Iterator<Item = &'a FinalizedExpr>>(
    state: &State,
    indent: usize,
    prefix: String,
    first_item: &str,
    expr_list: I,
) -> String {
    let mut result = prefix;

    result.push('{');
    result.push_str(first_item);

    for expr in expr_list {
        result.push('\n');
        result.push_str(&indent_to_string(indent));
        result.push_str(&expr.pretty_print(state, indent));
    }

    result.push('}');
    result
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, PartialOrd, Ord, Hash)]
pub struct InternedString(u32);

impl From<InternedString> for usize {
    fn from(t: InternedString) -> Self {
        t.0 as usize
    }
}

impl From<usize> for InternedString {
    fn from(t: usize) -> Self {
        InternedString(t as u32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LispExpr {
    Macro(LispMacro),
    Value(LispValue),
    OpVar(InternedString),
    Call(Vec<LispExpr>),
}

struct FinalizationContext {
    scope_level: Scope,
    arguments: Vec<(InternedString, (Scope, StackOffset, VariableConstraint))>,
    tail_call_status: TailCallStatus,
    own_name: Option<InternedString>,
}

impl FinalizationContext {
    fn new(own_name: Option<InternedString>) -> FinalizationContext {
        FinalizationContext {
            scope_level: Scope::default(),
            arguments: Vec::new(),
            tail_call_status: TailCallStatus::CanTailCall,
            own_name: own_name,
        }
    }
}

impl LispExpr {
    fn into_top_expr(self) -> EvaluationResult<TopExpr> {
        let is_define = if let LispExpr::Call(ref expr_list) = self {
            Some(&LispExpr::Macro(LispMacro::Define)) == expr_list.get(0)
        } else {
            false
        };

        // This feels kind of clumsy
        if is_define {
            match self {
                LispExpr::Call(expr_list) => {
                    let mut call_iter = expr_list.into_iter();
                    destructure!(call_iter, [_mac, opvar, definition], {
                        if let LispExpr::OpVar(n) = opvar {
                            Ok(TopExpr::Define(n, definition))
                        } else {
                            Err(EvaluationError::BadDefine)
                        }
                    })
                }
                _ => unreachable!(),
            }
        } else {
            Ok(TopExpr::Regular(
                self.finalize(&mut FinalizationContext::new(None))?.0,
            ))
        }
    }

    /// Bool indicates whether the expression returns
    /// Tail calls and recursions do not return, for example
    fn finalize(self, ctx: &mut FinalizationContext) -> EvaluationResult<(FinalizedExpr, bool)> {
        // TODO: better name
        fn deal_with_opvar(
            expr: LispExpr,
            ctx: &mut FinalizationContext,
            caller: Option<BuiltIn>,
        ) -> EvaluationResult<(FinalizedExpr, bool)> {
            if let LispExpr::OpVar(n) = expr {
                // So if we encounter a symbol, it could be two things:
                // a function argument, in which case it should be in the arguments map
                // a reference to something in our state.
                // Function arguments take precendence.
                Ok((
                    ctx.arguments
                        .iter_mut()
                        .rev()
                        .find(|&&mut (o, _)| o == n)
                        .map(|&mut (_, (arg_scope, arg_offset, ref mut move_status))| {
                            let replacement = match (*move_status, caller) {
                                (VariableConstraint::Unconstrained, Some(BuiltIn::Car)) => {
                                    VariableConstraint::RemovedHead
                                }
                                (VariableConstraint::Unconstrained, Some(BuiltIn::Cdr)) => {
                                    VariableConstraint::RemovedTail
                                }
                                _ => VariableConstraint::NeedFull,
                            };

                            FinalizedExpr::Argument(
                                arg_offset,
                                arg_scope,
                                replace(move_status, replacement),
                            )
                        })
                        .unwrap_or_else(|| FinalizedExpr::Variable(n)),
                    true,
                ))
            } else {
                expr.finalize(ctx)
            }
        }

        Ok(match self {
            LispExpr::Value(v) => (FinalizedExpr::Value(v), true),
            LispExpr::OpVar(..) => return deal_with_opvar(self, ctx, None),
            LispExpr::Macro(..) => {
                return Err(EvaluationError::UnexpectedOperator);
            }
            LispExpr::Call(expr_list) => {
                let mut expr_iter = expr_list.into_iter();
                let head_expr = match expr_iter.next() {
                    Some(head) => head,
                    None => return Err(EvaluationError::EmptyListEvaluation),
                };

                match head_expr {
                    LispExpr::Macro(LispMacro::Cond) => {
                        destructure!(expr_iter, [test_expr, true_expr, false_expr], {
                            let could_tail_call = ctx.tail_call_status;
                            let false_expr_args = ctx.arguments.clone();
                            let mut false_expr_ctx = FinalizationContext {
                                arguments: false_expr_args,
                                ..*ctx
                            };
                            let (finalized_false_expr, false_returns) =
                                false_expr.finalize(&mut false_expr_ctx)?;
                            let (finalized_true_expr, true_returns) = true_expr.finalize(ctx)?;

                            // Move analysis: a function argument is still moveable
                            // when it has been moved in neither the true branch or
                            // the false branch.
                            for (&mut (_, (_, _, ref mut arg_true)), &(_, (_, _, arg_false))) in
                                ctx.arguments
                                    .iter_mut()
                                    .zip(false_expr_ctx.arguments.iter())
                            {
                                *arg_true = arg_false.combine(*arg_true);
                            }

                            // The test expression cannot ever tail call!
                            // TODO: add test for this!
                            ctx.tail_call_status = TailCallStatus::CannotTailCall;

                            (
                                FinalizedExpr::Cond(
                                    Box::new((
                                        test_expr.finalize(ctx)?.0,
                                        finalized_true_expr,
                                        finalized_false_expr,
                                    )),
                                    false_returns,
                                    true_returns,
                                    could_tail_call,
                                ),
                                false_returns || true_returns,
                            )
                        })
                    }
                    LispExpr::Macro(LispMacro::Lambda) => {
                        destructure!(expr_iter, [arg_list, body], {
                            if let LispExpr::Call(ref arg_vec) = arg_list {
                                // Append arguments to the arguments map. Since we're doing
                                // symbol lookup in reverse orders, this guarantees that
                                // variables with the same symbol will use the highest
                                // scope.
                                let num_args = arg_vec.len();
                                let arguments_len = ctx.arguments.len();
                                ctx.arguments.reserve(num_args);

                                for (offset, expr) in arg_vec.into_iter().enumerate() {
                                    let symbol = match *expr {
                                        LispExpr::OpVar(intern) => Ok(intern),
                                        _ => Err(EvaluationError::MalformedDefinition),
                                    }?;

                                    ctx.arguments.push((
                                        symbol,
                                        (
                                            ctx.scope_level,
                                            StackOffset::from(offset),
                                            VariableConstraint::Unconstrained,
                                        ),
                                    ));
                                }

                                // Update context for lambda
                                let orig_scope_level = ctx.scope_level;
                                let current_tail_status = ctx.tail_call_status;
                                ctx.scope_level = ctx.scope_level.next();
                                ctx.tail_call_status = TailCallStatus::CanTailCall;

                                let finalized_body = body.finalize(ctx)?.0;

                                // TODO: here we can check whether this is not a tail call
                                // but all arguments were moved!

                                let result = FinalizedExpr::Lambda(
                                    num_args,
                                    orig_scope_level,
                                    Box::new(finalized_body),
                                );

                                // Reset context to original state
                                ctx.scope_level = orig_scope_level;
                                ctx.tail_call_status = current_tail_status;
                                ctx.arguments.truncate(arguments_len);

                                (result, true)
                            } else {
                                return Err(EvaluationError::ArgumentTypeMismatch);
                            }
                        })
                    }
                    // Defines should be caught by into_top_expr
                    LispExpr::Macro(LispMacro::Define) => {
                        return Err(EvaluationError::MalformedDefinition)
                    }
                    // Function evaluation
                    _ => {
                        let is_self_call = if let LispExpr::OpVar(intern) = head_expr {
                            ctx.own_name
                                .map(|self_name| intern == self_name)
                                .unwrap_or(false)
                        } else {
                            false
                        };
                        let is_tail_call = ctx.tail_call_status == TailCallStatus::CanTailCall;

                        ctx.tail_call_status = TailCallStatus::CannotTailCall;

                        // We traverse the arguments from last to first to make sure
                        // we get the argument moves correctly. The last arguments
                        // get to use the moves first.
                        let mut arg_finalized_expr = Vec::with_capacity(expr_iter.size_hint().0);
                        let caller = if let LispExpr::Value(
                            LispValue::Function(LispFunc::BuiltIn(builtin)),
                        ) = head_expr
                        {
                            Some(builtin)
                        } else {
                            None
                        };

                        for e in expr_iter.rev() {
                            let arg = deal_with_opvar(e, ctx, caller)?.0;
                            arg_finalized_expr.push(arg);
                        }
                        arg_finalized_expr.reverse();

                        let funk = head_expr.finalize(ctx)?.0;
                        (
                            FinalizedExpr::FunctionCall(
                                Box::new(funk),
                                arg_finalized_expr,
                                is_tail_call,
                                is_self_call,
                            ),
                            !is_tail_call || caller.is_some(),
                        )
                    }
                }
            }
        })
    }

    pub fn pretty_print(&self, state: &State, indent: usize) -> String {
        match *self {
            LispExpr::Value(ref v) => v.pretty_print(state, indent),
            LispExpr::OpVar(intern) => state.resolve_intern(intern).into(),
            LispExpr::Macro(ref mac) => format!("{:?}", mac),
            LispExpr::Call(ref expr_vec) => {
                let mut result = String::new();

                result.push('{');

                for (idx, expr) in expr_vec.iter().enumerate() {
                    if idx > 0 {
                        result.push('\n');
                        result.push_str(&indent_to_string(indent));
                    }

                    result.push_str(&expr.pretty_print(state, indent));
                }

                result.push('}');
                result
            }
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum VariableConstraint {
    NeedFull,
    Unconstrained,
    RemovedTail,
    RemovedHead,
}

impl VariableConstraint {
    fn combine(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            VariableConstraint::NeedFull
        }
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
    BadDefine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LispValue {
    Boolean(bool),
    Integer(u64),
    Function(LispFunc),
    List(Vec<LispValue>),
}

impl LispValue {
    fn get_type(&self) -> ArgType {
        match *self {
            LispValue::Boolean(..) => ArgType::Boolean,
            LispValue::Integer(..) => ArgType::Integer,
            LispValue::Function(..) => ArgType::Function,
            LispValue::List(..) => ArgType::List,
        }
    }

    pub fn pretty_print(&self, state: &State, indent: usize) -> String {
        match *self {
            LispValue::Function(ref func) => format!("[{}]", func.pretty_print(state, indent)),
            LispValue::Integer(i) => i.to_string(),
            LispValue::Boolean(true) => "#t".into(),
            LispValue::Boolean(false) => "#f".into(),
            LispValue::List(ref vec) => {
                let mut result = "(".to_string();

                for (idx, val) in vec.iter().enumerate() {
                    if idx > 0 {
                        result.push(' ');
                    }

                    result.push_str(&val.pretty_print(state, indent));
                }

                result.push(')');
                result
            }
        }
    }
}

fn builtin_instr(f: BuiltIn, arg_count: usize) -> EvaluationResult<Instr> {
    Ok(match (f, arg_count) {
        (BuiltIn::AddOne, 1) => Instr::AddOne,
        (BuiltIn::SubOne, 1) => Instr::SubOne,
        (BuiltIn::CheckZero, 1) => Instr::CheckZero,
        (BuiltIn::CheckNull, 1) => Instr::CheckNull,
        (BuiltIn::List, _) => Instr::List(arg_count),
        (BuiltIn::Cons, 2) => Instr::Cons,
        (BuiltIn::Car, 1) => Instr::Car,
        (BuiltIn::Cdr, 1) => Instr::Cdr,
        (BuiltIn::CheckType(t), 1) => Instr::CheckType(t),
        (_, _) => return Err(EvaluationError::ArgumentCountMismatch),
    })
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum VarStatus {
    RemovedHead,
    RemovedTail,
}

// Compiles a finalized expression into instructions and writes them to the
// given buffer *in reverse order*
fn inner_compile(
    expr: FinalizedExpr,
    state: &State,
    instructions: &mut Vec<Instr>,
    var_stats: &mut Vec<(StackOffset, Scope, VarStatus)>,
) -> EvaluationResult<()> {
    match expr {
        FinalizedExpr::Argument(offset, _scope, VariableConstraint::Unconstrained) => {
            instructions.push(Instr::MoveArgument(offset));
        }
        FinalizedExpr::Argument(offset, _scope, _move_status) => {
            instructions.push(Instr::CloneArgument(offset));
        }
        FinalizedExpr::Value(v) => {
            instructions.push(Instr::PushValue(v));
        }
        FinalizedExpr::Variable(n) => if let Some(v) = state.get(n) {
            instructions.push(Instr::PushValue(v.clone()));
        } else {
            return Err(EvaluationError::UnknownVariable(
                state.resolve_intern(n).into(),
            ));
        },
        FinalizedExpr::Cond(triple, false_expr_returns, true_expr_returns, tail_call_status) => {
            let unpacked = *triple;
            let (test, true_expr, false_expr) = unpacked;

            // Test must be done before to ensure that the true/ false branches
            // get the right var_stats.
            let mut test_expr_buf = Vec::new();
            inner_compile(test.clone(), state, &mut test_expr_buf, var_stats)?;
            let mut false_expr_var_stats = var_stats.clone();
            let before_len = instructions.len();

            if true_expr_returns && tail_call_status == TailCallStatus::CanTailCall {
                instructions.push(Instr::Return);
            }

            inner_compile(true_expr, state, instructions, var_stats)?;
            let jump_size = instructions.len() - before_len;
            let before_len = instructions.len();

            // If the false branch never returns, because it is a tail call,
            // we do not need to place a jump instruction
            if false_expr_returns {
                if tail_call_status == TailCallStatus::CanTailCall {
                    instructions.push(Instr::Return);
                } else {
                    instructions.push(Instr::Jump(jump_size));
                }
            }

            if let FinalizedExpr::FunctionCall(ref f_box, ref args, ..) = test {
                if let FinalizedExpr::Value(
                    LispValue::Function(LispFunc::BuiltIn(BuiltIn::CheckZero)),
                ) = **f_box
                {
                    if let Some(&FinalizedExpr::Argument(offset, scope, _)) = args.get(0) {
                        // OK, so at this point we know we are jumping conditionally
                        // on whether a function arg is zero.
                        // Next: make sure that every use of this argument in the false branch
                        // is within a sub1 call.
                        // If this is the case, replace all these sub1 calls by uses
                        // of the argument itself (maintaining its move status!).
                        // Then, encode the conditional jump, zero check and decrement using
                        // a single, superspecialized instruction.
                        if args.len() == 1 && false_expr.only_use_after_sub(offset, scope, false) {
                            let new_false_expr = false_expr.remove_subs_of(offset, scope);
                            inner_compile(
                                new_false_expr,
                                state,
                                instructions,
                                &mut false_expr_var_stats,
                            )?;
                            let jump_size = instructions.len() - before_len;
                            instructions.push(Instr::CondZeroJumpDecr(offset, jump_size));

                            return Ok(());
                        }
                    }
                }
            }

            inner_compile(false_expr, state, instructions, &mut false_expr_var_stats)?;
            let jump_size = instructions.len() - before_len;
            instructions.push(Instr::CondJump(jump_size));
            instructions.extend(test_expr_buf.into_iter());
        }
        FinalizedExpr::Lambda(arg_count, scope, body) => {
            instructions.push(Instr::CreateLambda(scope, arg_count, body));
        }
        FinalizedExpr::FunctionCall(funk, args, is_tail_call, is_self_call) => {
            // Here we check for special patterns of builtin functions on single
            // arguments and try to generate specialized instructions for them.
            if let FinalizedExpr::Value(LispValue::Function(LispFunc::BuiltIn(bf))) = *funk {
                if let Some(&FinalizedExpr::Argument(offset, scope, move_status)) = args.get(0) {
                    match (bf, offset, scope, move_status) {
                        (BuiltIn::Car, offset, scope, VariableConstraint::RemovedTail) => {
                            instructions.push(Instr::VarSplit(offset));
                            var_stats.push((offset, scope, VarStatus::RemovedHead));
                            return Ok(());
                        }
                        (BuiltIn::Cdr, offset, scope, VariableConstraint::Unconstrained) => {
                            if var_stats.iter().any(|&(o, s, v)| {
                                o == offset && s == scope && v == VarStatus::RemovedHead
                            }) {
                                // Head was previously removed. We can just move the remainder
                                instructions.push(Instr::MoveArgument(offset));
                                return Ok(());
                            }
                        }
                        (BuiltIn::Cdr, offset, scope, VariableConstraint::RemovedHead) => {
                            instructions.push(Instr::VarReverseSplit(offset));
                            var_stats.push((offset, scope, VarStatus::RemovedTail));
                            return Ok(());
                        }
                        (BuiltIn::Car, offset, scope, VariableConstraint::Unconstrained) => {
                            if var_stats.iter().any(|&(o, s, v)| {
                                o == offset && s == scope && v == VarStatus::RemovedTail
                            }) {
                                // Tail was previously removed. We can just move the head
                                instructions.push(Instr::MoveArgument(offset));
                            } else {
                                // Tail is still on.
                                instructions.push(Instr::VarCar(offset));
                            }
                            return Ok(());
                        }
                        (BuiltIn::Car, offset, ..) |
                        (BuiltIn::CheckNull, offset, ..) |
                        (BuiltIn::CheckZero, offset, ..) => {
                            // Inspection mode!
                            instructions.push(match bf {
                                BuiltIn::CheckNull => Instr::VarCheckNull(offset),
                                BuiltIn::CheckZero => Instr::VarCheckZero(offset),
                                BuiltIn::Car => Instr::VarCar(offset),
                                _ => unreachable!(),
                            });
                            return Ok(());
                        }
                        (BuiltIn::AddOne, offset, _scope, VariableConstraint::Unconstrained) => {
                            instructions.push(Instr::MoveArgument(offset));
                            instructions.push(Instr::VarAddOne(offset));
                            return Ok(());
                        }
                        (..) => {}
                    }
                }
            }

            // Here, for tail calls, we try to reuse function arguments
            // and elide copies thereof. For strict recursions, it's
            // possible to do a (partial) elision when some non-zero prefix
            // of the called function arguments is an in-order move from the
            // prefix of the calling function.
            let args_len = args.len();
            let init_len = instructions.len();
            let builtin =
                if let FinalizedExpr::Value(LispValue::Function(LispFunc::BuiltIn(bf))) = *funk {
                    Some(bf)
                } else {
                    None
                };

            if let Some(bf) = builtin {
                instructions.push(builtin_instr(bf, args_len)?);
            } else if is_self_call && is_tail_call {
                instructions.push(Instr::Recurse(args_len));
            } else {
                instructions.push(Instr::EvalFunction(args_len, None));
                inner_compile(*funk, state, instructions, var_stats)?;
            }

            // Compiling all arguments
            let arg_instr_vecs: Vec<_> = args.into_iter()
                .map(|expr| {
                    let mut sub_buf = Vec::new();
                    inner_compile(expr, state, &mut sub_buf, var_stats)?;
                    Ok(sub_buf)
                })
                .collect::<Result<_, _>>()?;

            let arg_skip_count = arg_instr_vecs
                .iter()
                .enumerate()
                .take_while(|&(idx, buf): &(_, &Vec<_>)| {
                    if let Instr::MoveArgument(offset) = *buf.index(0) {
                        idx == offset.to_usize()
                    } else {
                        false
                    }
                })
                .count();

            for (idx, mut buf) in arg_instr_vecs.into_iter().enumerate().rev() {
                if idx < arg_skip_count && is_tail_call {
                    instructions.extend(buf.drain(1..));
                } else {
                    instructions.extend(buf.into_iter());
                }
            }

            if is_self_call && is_tail_call {
                // Store the number of copies that we have to be for
                // execution time.
                instructions[init_len] = Instr::Recurse(args_len - arg_skip_count);
            } else if is_tail_call && !builtin.is_some() {
                instructions[init_len] = Instr::EvalFunction(args_len, Some(arg_skip_count));
            }
        }
    }

    Ok(())
}

fn compile_finalized_expr(expr: FinalizedExpr, state: &State) -> EvaluationResult<Vec<Instr>> {
    let mut instructions = Vec::with_capacity(32);
    //instructions.push(Instr::Return);

    inner_compile(expr, state, &mut instructions, &mut Vec::new())?;

    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::parse::{parse_lisp_string, ParseError};
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

    fn check_lisp<'i, I>(state: &mut State, commands: I) -> Result<LispValue, LispError>
    where
        I: IntoIterator<Item = &'i str>,
    {
        let mut last_ret_val = None;

        for cmd in commands {
            let expr = parse_lisp_string(cmd, state)?;
            last_ret_val = Some(evaluator::eval(expr, state)?);
        }

        Ok(last_ret_val.unwrap())
    }

    fn check_lisp_ok<'i, I>(commands: I, expected_out: &str)
    where
        I: IntoIterator<Item = &'i str>,
    {
        let mut state = State::default();
        let val = check_lisp(&mut state, commands).unwrap();
        assert_eq!(expected_out, val.pretty_print(&state, 0));
    }

    fn check_lisp_err<'i, I>(commands: I, expected_err: LispError)
    where
        I: IntoIterator<Item = &'i str>,
    {
        let mut state = State::default();
        assert_eq!(expected_err, check_lisp(&mut state, commands).unwrap_err());
    }

    fn get_bytecode(definition: &str, self_name: &str) -> Vec<Instr> {
        let mut state = State::default();
        check_lisp(
            &mut state,
            vec![&format!("(define {} 1337)", self_name)[..]],
        ).unwrap();
        let intern = state.intern(self_name);
        let expr = super::parse::parse_lisp_string(definition, &mut state).unwrap();
        let mut finalization_ctx = super::FinalizationContext::new(Some(intern));
        let finalized_expr = expr.finalize(&mut finalization_ctx).unwrap().0;

        if let FinalizedExpr::Lambda(.., body) = finalized_expr {
            super::compile_finalized_expr(*body, &mut state).unwrap()
        } else {
            super::compile_finalized_expr(finalized_expr, &mut state).unwrap()
        }
    }

    #[test]
    fn add_bytecode() {
        let bytecode = get_bytecode(
            "(lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y))))",
            "add",
        );

        assert_eq!(
            bytecode,
            vec![
                Instr::MoveArgument(From::from(0)),
                Instr::Recurse(0),
                Instr::VarAddOne(From::from(0)),
                Instr::CondZeroJumpDecr(From::from(1), 2),
            ]
        );
    }

    #[test]
    fn map_bytecode() {
        let bytecode = get_bytecode(
            "(lambda (f xs) (cond (null? xs) xs (cons (f (car xs)) (map f (cdr xs)))))",
            "map",
        );

        assert_eq!(
            bytecode,
            vec![
                Instr::MoveArgument(From::from(1)),
                Instr::Jump(1),
                Instr::Cons,
                Instr::EvalFunction(2, None),
                // 1337 is the magic number representing the function itself
                Instr::PushValue(LispValue::Integer(1337)),
                Instr::MoveArgument(From::from(1)),
                Instr::MoveArgument(From::from(0)),
                Instr::EvalFunction(1, None),
                Instr::CloneArgument(From::from(0)),
                Instr::VarSplit(From::from(1)),
                Instr::CondJump(9),
                Instr::VarCheckNull(From::from(1)),
            ]
        );
    }

    #[test]
    fn comp_bytecode() {
        let bytecode = get_bytecode(
            "(lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y)))))",
            ">",
        );
        assert_eq!(
            bytecode,
            vec![
                Instr::PushValue(LispValue::Boolean(false)),
                Instr::Jump(1),
                Instr::PushValue(LispValue::Boolean(true)),
                Instr::Recurse(0),
                Instr::CondZeroJumpDecr(From::from(1), 1),
                Instr::CondZeroJumpDecr(From::from(0), 4),
            ]
        );
    }

    // TODO: add test for partial copy recursive functions.

    #[test]
    fn shadowing() {
        check_lisp_ok(
            vec![
                "(define f (lambda (x) (lambda (x) x)))",
                "(list ((f #t) 3) ((f (list)) #f))",
            ],
            "(3 #f)",
        );
    }

    #[test]
    fn cond_argument() {
        check_lisp_ok(
            vec![
                "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(add (cond (> 12 5) 1 2) 3)",
            ],
            "4",
        );
    }

    #[test]
    fn shadowing_two() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "((lambda (x) (add ((lambda (x) (add1 x)) x) x)) 10)",
            ],
            "21",
        );
    }

    #[test]
    fn shadowing_three() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "((lambda (x) (add x ((lambda (x) (add (sub1 x) x)) x))) 4)",
            ],
            "11",
        );
    }

    #[test]
    fn display_int_val() {
        let state = Default::default();
        let val = LispValue::Integer(5);
        assert_eq!("5", val.pretty_print(&state, 0));
    }

    #[test]
    fn display_list_val() {
        let state = Default::default();
        let val = LispValue::List(vec![LispValue::Integer(1), LispValue::List(vec![])]);
        assert_eq!("(1 ())", val.pretty_print(&state, 0));
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
    fn map2_zip() {
        check_lisp_ok(
            vec![
                "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                "(define or (lambda (x y) (cond x #t y)))",
                "(define zip (lambda (x y) (cond (or (null? x) (null? y)) (list) (cons (list (car x) (car y)) (zip (cdr x) (cdr y))))))",
                "(define map2 (lambda (f l) (cond (null? l) (list) (cons (f (car (cdr (car l))) (car (car l))) (map2 f (cdr l))))))",
                "(map2 add (zip (list 1 2 3 4 5) (list 0 20 40 60)))",
            ],
            "(2 23 44 65)",
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
    fn variable_overwrite() {
        check_lisp_err(
            vec!["(define x 1)", "(define x 1000)", "(add1 x)"],
            LispError::Evaluation(EvaluationError::BadDefine),
        );
    }

    #[test]
    fn eval_empty_call() {
        check_lisp_err(
            vec!["()"],
            LispError::Evaluation(EvaluationError::EmptyListEvaluation),
        );
    }

    #[test]
    fn check_int() {
        check_lisp_ok(
            vec![
                "(define map (lambda (f xs) (cond (null? xs) (list) (cons (f (car xs)) (map f (cdr xs))))))",
                "(map int? (list 1 2 #t (list 3 4) add1 0))",
            ],
            "(#t #t #f #f #f #t)",
        );
    }

    #[test]
    fn check_bool() {
        check_lisp_ok(
            vec![
                "(list (bool? #t) (bool? #f) (bool? 5) (bool? (list #t)) (bool? bool?))",
            ],
            "(#t #t #f #f #f)",
        );
    }

    #[test]
    fn check_fun() {
        check_lisp_ok(
            vec![
                "(list (fun? #t) (fun? fun?) (fun? 0) (fun? bool?) (fun? (lambda (x) #t)) (fun? #t))",
            ],
            "(#f #t #f #t #t #f)",
        );
    }

    #[test]
    fn check_use_var_after_cond_zero() {
        check_lisp_ok(
            vec!["(define f (lambda (x) (cond (zero? x) x x)))", "(f 1)"],
            "1",
        );
    }

    #[test]
    fn check_list() {
        check_lisp_ok(
            vec!["(list (list? list) (list? list?) (list? (list)))"],
            "(#f #f #t)",
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
    fn lower_level_define() {
        check_lisp_err(
            vec!["(list (define x 5))"],
            LispError::Evaluation(EvaluationError::MalformedDefinition),
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
    fn non_function_app() {
        check_lisp_err(
            vec!["(10 3)"],
            LispError::Evaluation(EvaluationError::NonFunctionApplication),
        );
    }

    #[test]
    fn unexpected_operator() {
        check_lisp_err(
            vec!["(cond cond cond cond)"],
            LispError::Evaluation(EvaluationError::UnexpectedOperator),
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
    fn popn() {
        check_lisp_ok(
            vec![
                "(define popn (lambda (l n) (cond (zero? n) l (popn (cdr l) (sub1 n)))))",
                "(popn (list 1 2 3 4 5) 2)",
            ],
            "(1 2 3)",
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

    const SORT_COMMANDS: &[&str] = &[
        "(define filter (lambda (f xs) (cond (null? xs) (list) (cond (f (car xs)) (cons (car xs) (filter f (cdr xs))) (filter f (cdr xs))))))",
        "(define not (lambda (t) (cond t #f #t)))",
        "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
        "(define and (lambda (t1 t2) (cond t1 t2 #f)))",
        "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
        "(define sort (lambda (l) (cond (null? l) l (append (cons (car l) (sort (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort (filter (lambda (x) (> x (car l))) l))))))",
        "(define sort_rev (lambda (l) (cond (null? l) l (app_rev (cons (car l) (sort_rev (filter (lambda (x) (not (< x (car l)))) (cdr l)))) (sort' (filter (lambda (x) (< x (car l))) l))))))",
        "(define sort' (lambda (l) (cond (null? l) l (app_rev (cons (car l) (sort' (filter (lambda (x) (not (> x (car l)))) (cdr l)))) (sort_rev (filter (lambda (x) (> x (car l))) l))))))",
        "(define app_rev (lambda (l r) (cond (null? r) l (app_rev (cons (car r) l) (cdr r)))))",
        "(define < (lambda (x y) (> y x)))",
    ];

    #[test]
    fn sort() {
        check_lisp_ok(
            SORT_COMMANDS
                .into_iter()
                .cloned()
                .chain(Some("(sort (list 5 3 2 10 0 7))").into_iter()),
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
        let mut state = State::default();
        assert!(check_lisp(&mut state, vec!["(list add1 ((lambda (f x) (f x)) sub1))"]).is_ok());
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
    fn cyclic_func_calls() {
        check_lisp_ok(
            vec![
                "(define <' (lambda (x y) (cond (zero? y) #f (< x (sub1 y)))))",
                "(define < (lambda (x y) (cond (zero? x) (cond (zero? y) #f #t) (<' (sub1 x) y))))",
                "(list (< 1 2) (< 0 1) (< 2 2) (< 1 1) (< 1 0) (< 2 1))",
            ],
            "(#t #t #f #f #f #f)",
        );
    }

    // TODO: add test for non-copying TCO

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
    fn bench_add_including_compilation(b: &mut super::test::Bencher) {
        b.iter(|| {
            let mut state = State::default();
            check_lisp(
                &mut state,
                vec![
                    "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                    "(add 100 100)",
                ],
            )
        });
    }

    #[bench]
    fn bench_curry(b: &mut super::test::Bencher) {
        let mut state = State::default();
        let init_commands = vec![
            "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
            "(define curried-add (lambda (x y) ((add x) y)))",
        ];

        for cmd in init_commands {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| check_lisp(&mut state, vec!["(curried-add 100 100)"]));
    }

    #[bench]
    fn bench_mutual_recursion(b: &mut super::test::Bencher) {
        let mut state = State::default();
        let init_commands = vec![
            "(define <' (lambda (x y) (cond (zero? y) #f (< x (sub1 y)))))",
            "(define < (lambda (x y) (cond (zero? x) (cond (zero? y) #f #t) (<' (sub1 x) y))))",
        ];

        for cmd in init_commands {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| check_lisp(&mut state, vec!["(< 10000 10000)"]));
    }

    #[bench]
    fn bench_big_add(b: &mut super::test::Bencher) {
        let mut state = State::default();
        let init_commands = vec![
            "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
        ];

        for cmd in init_commands {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| check_lisp(&mut state, vec!["(add 100000 100000)"]));
    }

    /// Benchmarks list intensive code
    #[bench]
    fn bench_arithmetic_sums(b: &mut super::test::Bencher) {
        let mut state = State::default();
        let init_commands = vec![
            "(define > (lambda (x y) (cond (zero? x) #f (cond (zero? y) #t (> (sub1 x) (sub1 y))))))",
            "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
            "(define range (lambda (start end) (cond (> end start) (cons end (range start (sub1 end))) (list start))))",
            "(define map2 (lambda (f l) (cond (null? l) l (cons (f (car (cdr (car l))) (car (car l))) (map2 f (cdr l))))))",
            "(define foldr (lambda (xs f init) (cond (null? xs) init (foldr (cdr xs) f (f init (car xs))))))",
            "(define zip (lambda (x y) (cond (or (null? x) (null? y)) (list) (cons (list (car x) (car y)) (zip (cdr x) (cdr y))))))",
            "(define reverse (lambda (l) (cond (null? l) l (append (list (car l)) (reverse (cdr l))))))",
            "(define append (lambda (l1 l2) (cond (null? l2) l1 (cons (car l2) (append l1 (cdr l2))))))",
            "(define or (lambda (x y) (cond x #t y)))",
        ];

        for cmd in init_commands {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| {
            let expr = parse_lisp_string(
                "(foldr (map2 add (zip (range 1 50) (reverse (range 1 50)))) add 0)",
                &mut state,
            ).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        });
    }

    #[bench]
    fn bench_mult(b: &mut super::test::Bencher) {
        b.iter(|| {
            let mut state = State::default();
            check_lisp(
                &mut state,
                vec![
                    "(define add (lambda (x y) (cond (zero? y) x (add (add1 x) (sub1 y)))))",
                    "(define mult (lambda (x y) (cond (zero? y) 0 (add (mult x (sub1 y)) x))))",
                    "(mult 10 100)",
                ],
            )
        });
    }

    #[bench]
    fn bench_sort(b: &mut super::test::Bencher) {
        let mut state = State::default();

        for cmd in SORT_COMMANDS {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| {
            let expr = parse_lisp_string(
                "(sort (list 25 3 40 5 1 0 3 2 10 30 0 7 1 2 300 5 3 13 3 0 1 2 2 3 1))",
                &mut state,
            ).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        });
    }

    #[bench]
    fn bench_sort_alt(b: &mut super::test::Bencher) {
        let mut state = State::default();

        for cmd in SORT_COMMANDS {
            let expr = parse_lisp_string(cmd, &mut state).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        }

        b.iter(|| {
            let expr = parse_lisp_string(
                "(sort' (list 25 3 40 5 1 0 3 2 10 30 0 7 1 2 300 5 3 13 3 0 1 2 2 3 1))",
                &mut state,
            ).unwrap();
            evaluator::eval(expr, &mut state).unwrap();
        });
    }
}
