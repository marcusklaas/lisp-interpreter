use super::{ArgType, BuiltIn, CustomFunc, LispExpr, LispFunc, LispMacro, LispValue};
use super::evaluator::State;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use petgraph::Undirected;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::{Config, Dot};
use std::fmt;

#[derive(Clone)]
struct FunctionReference {
    ins: Vec<NodeIndex>,
    out: NodeIndex,
}

#[derive(Debug)]
pub enum GeneralizedExpr<'e> {
    Expr(&'e LispExpr),
    DumbNode(String),
}

impl<'e> fmt::Display for GeneralizedExpr<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GeneralizedExpr::Expr(ref e) => e.fmt(f),
            GeneralizedExpr::DumbNode(ref s) => write!(f, "{}", s),
        }
    }
}

struct Context<'e> {
    graph: Graph<GeneralizedExpr<'e>, NoLabel, Undirected>,
    reference_map: HashMap<LispFunc, FunctionReference>,
    list_index: NodeIndex,
    // wrapped_index: NodeIndex,
    bool_index: NodeIndex,
    int_index: NodeIndex,
    state: &'e State,
}

impl<'e> Context<'e>{
    fn new_from_state(state: &'e State) -> Self {
        let mut graph = Graph::<GeneralizedExpr, NoLabel, Undirected>::new_undirected();
        let bool_idx = graph.add_node(GeneralizedExpr::DumbNode("Boolean".to_owned()));
        let int_idx = graph.add_node(GeneralizedExpr::DumbNode("Integer".to_owned()));
        let list_idx = graph.add_node(GeneralizedExpr::DumbNode("List".to_owned()));
        // let wrapped_idx = graph.add_node(GeneralizedExpr::DumbNode("Wrapped".to_owned()));
        let map = create_ref_map(&mut graph, bool_idx, int_idx, list_idx /* , wrapped_idx */);

        Context {
            graph: graph,
            reference_map: map,
            // wrapped_index: wrapped_idx,
            list_index: list_idx,
            bool_index: bool_idx,
            int_index: int_idx,
            state: state,
        }
    }
}

pub struct NoLabel;

impl fmt::Display for NoLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

fn create_ref_map(
    graph: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
    list_index: NodeIndex,
    // wrapped_index: NodeIndex,
) -> HashMap<LispFunc, FunctionReference> {

    [
        (BuiltIn::AddOne, int_index, int_index),
        (BuiltIn::SubOne, int_index, int_index),
        (BuiltIn::CheckZero, int_index, bool_index),
        (BuiltIn::Cdr, list_index, list_index),
        // (BuiltIn::Car, list_index, wrapped_index),
        (BuiltIn::CheckNull, list_index, bool_index),
    ].into_iter()
        .map(|&(builtin, in_index, out_index)| {
            let in_node = graph.add_node(GeneralizedExpr::DumbNode(format!("{} in", builtin)));
            let out_node = graph.add_node(GeneralizedExpr::DumbNode(format!("{} out", builtin)));
            graph.add_edge(in_node, in_index, NoLabel);
            graph.add_edge(out_node, out_index, NoLabel);
            (
                LispFunc::BuiltIn(builtin),
                FunctionReference {
                    ins: vec![in_node],
                    out: out_node,
                },
            )
        })
        .collect()
}

// TODO: we should actually take the arguments instead of their types
//       for when a function is passed in.
pub fn make_specialization_graph<'e>(
    f: &'e CustomFunc,
    argument_types: &[ArgType],
    state: &'e State,
) -> Result<Graph<GeneralizedExpr<'e>, NoLabel, Undirected>, ()> {
    let mut context = Context::new_from_state(state);

    let main_ref = add_custom_func(f, &mut context).clone();
    for (&arg_node, &arg_type) in main_ref.ins.iter().zip(argument_types.iter()) {
        match arg_type {
            ArgType::Boolean => {
                context.graph.add_edge(arg_node, context.bool_index, NoLabel);
            }
            ArgType::Integer => {
                context.graph.add_edge(arg_node, context.int_index, NoLabel);
            }
            ArgType::List => {
                context.graph.add_edge(arg_node, context.list_index, NoLabel);
            }
            _ => {}
        }
    }

    let res = expand_graph(
        &(*f.body),
        &main_ref,
        &mut context
    )?;
    context.graph.add_edge(res, main_ref.out, NoLabel);

    println!("{}", Dot::with_config(&context.graph, &[Config::EdgeNoLabel]));

    Ok(context.graph)
}

fn add_custom_func<'m>(
    f: & CustomFunc,
    context: &'m mut Context,
) -> &'m FunctionReference {
    let arg_nodes = (0..f.arg_count)
        .map(|i| {
            context.graph.add_node(GeneralizedExpr::DumbNode(format!("in {}", i)))
        })
        .collect::<Vec<_>>();
    let out_node = context.graph.add_node(GeneralizedExpr::DumbNode("out".to_owned()));

    let reference = FunctionReference {
        ins: arg_nodes,
        out: out_node,
    };

    // TODO: don't clone, somehow
    if let Entry::Vacant(vacant) =
        context.reference_map.entry(LispFunc::Custom(f.clone()))
    {
        vacant.insert(reference)
    } else {
        unreachable!()
    }
}

fn eval_custom_func<'m, 'e: 'm>(
    val: &'e LispValue,
    self_node: NodeIndex,
    arg_indices: Vec<NodeIndex>,
    tail_len: usize,
    context: &'m mut Context<'e>,
) -> Result<NodeIndex, ()> {
    match *val {
        LispValue::Function(ref f) => {
            if let Some(reference) = context.reference_map.get(f) {
                // Let's not do crazy currying stuff yet and only
                // accept the exact number of arguments required.
                if reference.ins.len() != tail_len {
                    return Err(());
                } else {
                    for (&idx, &function_in) in arg_indices.iter().zip(reference.ins.iter()) {
                        context.graph.add_edge(idx, function_in, NoLabel);
                    }

                    context.graph.add_edge(self_node, reference.out, NoLabel);
                    return Ok(self_node);
                }
            }

            if let LispFunc::Custom(ref custom_func) = *f {
                let new_ref = add_custom_func(custom_func, context).clone();

                expand_graph(
                    &(*custom_func.body),
                    &new_ref,
                    context,
                )?;

                context.graph.add_edge(self_node, new_ref.out, NoLabel);
                Ok(self_node)
            } else {
                // Unfit built-in
                Err(())
            }
        }
        _ => {
            // Non function application
            Err(())
        }
    }
}

fn expand_graph<'m, 'e: 'm>(
    expr: &'e LispExpr,
    main_ref: &FunctionReference,
    context: &'m mut Context<'e>,
) -> Result<NodeIndex, ()> {
    match *expr {
        LispExpr::Argument(index, _c) => Ok(main_ref.ins[index]),
        LispExpr::Call(ref arg_vec, _is_tail_call, is_self_call) => {
            let self_node = context.graph.add_node(GeneralizedExpr::Expr(expr));

            if let [ref head, ref tail..] = arg_vec[..] {
                let arg_indices = tail.iter()
                    .map(|arg_expr| {
                        expand_graph(
                            arg_expr,
                            main_ref,
                            context,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                match *head {
                    // Recursion
                    _ if is_self_call => {
                        // Not the right number of arguments for recursion
                        if arg_indices.len() != main_ref.ins.len() {
                            return Err(());
                        }

                        for (&top_node, &arg_node) in main_ref.ins.iter().zip(arg_indices.iter()) {
                            context.graph.add_edge(top_node, arg_node, NoLabel);
                        }

                        context.graph.add_edge(self_node, main_ref.out, NoLabel);
                    }
                    LispExpr::Macro(m) => {
                        match m {
                            LispMacro::Cond => {
                                if tail.len() != 3 {
                                    return Err(());
                                }

                                context.graph.add_edge(arg_indices[0], context.bool_index, NoLabel);
                                context.graph.add_edge(self_node, arg_indices[1], NoLabel);
                                context.graph.add_edge(self_node, arg_indices[2], NoLabel);
                            }
                            // Do not bother with definitions/ lambdas (for now)
                            LispMacro::Define | LispMacro::Lambda => return Err(()),
                        }
                    }
                    LispExpr::Value(ref val) => {
                        return eval_custom_func(
                            val,
                            self_node,
                            arg_indices,
                            tail.len(),
                            context,
                        );
                    }
                    LispExpr::OpVar(ref n) => if let Some(state_index) = context.state.get_index(n) {
                        return eval_custom_func(
                            &context.state[state_index],
                            self_node,
                            arg_indices,
                            tail.len(),
                            context,
                        );
                    } else {
                        return Err(());
                    },
                    ref x => {
                        // TODO: implement this. I think it is necessary for expressions
                        // like ((add 1) 10)
                        return Ok(context.graph.add_node(GeneralizedExpr::DumbNode(
                            format!("non-value function head: {}", x),
                        )))
                    }
                }
            } else {
                // Empty call
                return Err(());
            }

            Ok(self_node)
        }
        LispExpr::Value(ref val) => {
            let general_expr: GeneralizedExpr = GeneralizedExpr::Expr(expr);
            let self_node = context.graph.add_node(general_expr);

            match *val {
                LispValue::Boolean(..) => {
                    context.graph.add_edge(self_node, context.bool_index, NoLabel);
                }
                LispValue::Integer(..) => {
                    context.graph.add_edge(self_node, context.int_index, NoLabel);
                }
                LispValue::Function(ref _f) => {
                    // TODO: implement function stuff?
                    return Ok(context.graph.add_node(GeneralizedExpr::DumbNode("funkkk".to_owned())));
                }
                LispValue::List(..) => return Err(()),
            }

            Ok(self_node)
        }
        // All references should be resolved at this point, right?
        // No, not per se.
        LispExpr::OpVar(..) => unreachable!(),
        // We shouldn't find a top level macro - this is almost surely a faulty
        // expression.
        LispExpr::Macro(_) => return Err(()),
    }
}
