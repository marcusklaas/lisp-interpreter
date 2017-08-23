use super::{ArgType, BuiltIn, CustomFunc, LispExpr, LispFunc, LispMacro, LispValue};
use super::evaluator::State;
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
pub enum GeneralizedExpr {
    Expr(LispExpr),
    DumbNode(String),
}

impl fmt::Display for GeneralizedExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GeneralizedExpr::Expr(ref e) => e.fmt(f),
            GeneralizedExpr::DumbNode(ref s) => write!(f, "{}", s),
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
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
) -> HashMap<LispFunc, FunctionReference> {

    [
        (BuiltIn::AddOne, int_index, int_index),
        (BuiltIn::SubOne, int_index, int_index),
        (BuiltIn::CheckZero, int_index, bool_index),
    ].into_iter()
        .map(|&(builtin, in_index, out_index)| {
            let in_node = g.add_node(GeneralizedExpr::DumbNode(format!("{} in", builtin)));
            let out_node = g.add_node(GeneralizedExpr::DumbNode(format!("{} out", builtin)));
            g.add_edge(in_node, in_index, NoLabel);
            g.add_edge(out_node, out_index, NoLabel);
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
pub fn make_specialization_graph(
    f: CustomFunc,
    argument_types: &[ArgType],
    state: &State,
) -> Result<Graph<GeneralizedExpr, NoLabel, Undirected>, ()> {
    let mut g = Graph::<GeneralizedExpr, NoLabel, Undirected>::new_undirected();
    let bool_idx = g.add_node(GeneralizedExpr::DumbNode("Boolean".to_owned()));
    let int_idx = g.add_node(GeneralizedExpr::DumbNode("Integer".to_owned()));
    let mut map = create_ref_map(&mut g, bool_idx, int_idx);

    let main_ref = add_custom_func(f.clone(), &mut g, &mut map);
    for (&arg_node, &arg_type) in main_ref.ins.iter().zip(argument_types.iter()) {
        match arg_type {
            ArgType::Boolean => {
                g.add_edge(arg_node, bool_idx, NoLabel);
            }
            ArgType::Integer => {
                g.add_edge(arg_node, int_idx, NoLabel);
            }
            _ => {}
        }
    }

    let res = expand_graph(
        &(*f.body),
        &main_ref,
        &mut g,
        &mut map,
        bool_idx,
        int_idx,
        state,
    )?;
    g.add_edge(res, main_ref.out, NoLabel);

    println!("{}", Dot::with_config(&g, &[Config::EdgeNoLabel]));

    Ok(g)
}

fn add_custom_func(
    f: CustomFunc,
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    reference_map: &mut HashMap<LispFunc, FunctionReference>,
) -> FunctionReference {
    let arg_nodes = (0..f.arg_count)
        .map(|i| {
            g.add_node(GeneralizedExpr::DumbNode(format!("in {}", i)))
        })
        .collect::<Vec<_>>();
    let out_node = g.add_node(GeneralizedExpr::DumbNode("out".to_owned()));

    let reference = FunctionReference {
        ins: arg_nodes,
        out: out_node,
    };

    reference_map.insert(LispFunc::Custom(f), reference.clone());

    reference
}

fn eval_custom_func(
    val: &LispValue,
    self_node: NodeIndex,
    arg_indices: Vec<NodeIndex>,
    tail_len: usize,
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    reference_map: &mut HashMap<LispFunc, FunctionReference>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
    state: &State,
) -> Result<NodeIndex, ()> {
    match *val {
        LispValue::Function(ref f) => {
            if let Some(reference) = reference_map.get(f) {
                // Let's not do crazy currying stuff yet and only
                // accept the exact number of arguments required.
                if reference.ins.len() != tail_len {
                    return Err(());
                } else {
                    for (&idx, &function_in) in arg_indices.iter().zip(reference.ins.iter()) {
                        g.add_edge(idx, function_in, NoLabel);
                    }

                    g.add_edge(self_node, reference.out, NoLabel);
                    return Ok(self_node);
                }
            }

            if let LispFunc::Custom(ref custom_func) = *f {
                let new_ref = add_custom_func(custom_func.clone(), g, reference_map);

                expand_graph(
                    &(*custom_func.body),
                    &new_ref,
                    g,
                    reference_map,
                    bool_index,
                    int_index,
                    state,
                )?;

                g.add_edge(self_node, new_ref.out, NoLabel);
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

fn expand_graph(
    expr: &LispExpr,
    main_ref: &FunctionReference,
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    reference_map: &mut HashMap<LispFunc, FunctionReference>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
    state: &State,
) -> Result<NodeIndex, ()> {
    match *expr {
        LispExpr::Argument(index, _c) => Ok(main_ref.ins[index]),
        LispExpr::Call(ref arg_vec, _is_tail_call, is_self_call) => {
            let self_node = g.add_node(GeneralizedExpr::Expr(expr.clone()));

            if let [ref head, ref tail..] = arg_vec[..] {
                let arg_indices = tail.iter()
                    .map(|arg_expr| {
                        expand_graph(
                            arg_expr,
                            main_ref,
                            g,
                            reference_map,
                            bool_index,
                            int_index,
                            state,
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
                            g.add_edge(top_node, arg_node, NoLabel);
                        }

                        g.add_edge(self_node, main_ref.out, NoLabel);
                    }
                    LispExpr::Macro(m) => {
                        match m {
                            LispMacro::Cond => {
                                if tail.len() != 3 {
                                    return Err(());
                                }

                                g.add_edge(arg_indices[0], bool_index, NoLabel);
                                g.add_edge(self_node, arg_indices[1], NoLabel);
                                g.add_edge(self_node, arg_indices[2], NoLabel);
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
                            g,
                            reference_map,
                            bool_index,
                            int_index,
                            state,
                        );
                    }
                    LispExpr::OpVar(ref n) => if let Some(state_index) = state.get_index(n) {
                        return eval_custom_func(
                            &state[state_index],
                            self_node,
                            arg_indices,
                            tail.len(),
                            g,
                            reference_map,
                            bool_index,
                            int_index,
                            state,
                        );
                    } else {
                        return Err(());
                    },
                    ref x => {
                        return Ok(g.add_node(GeneralizedExpr::DumbNode(
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
            let self_node = g.add_node(GeneralizedExpr::Expr(expr.clone()));

            match *val {
                LispValue::Boolean(..) => {
                    g.add_edge(self_node, bool_index, NoLabel);
                }
                LispValue::Integer(..) => {
                    g.add_edge(self_node, int_index, NoLabel);
                }
                LispValue::Function(ref _f) => {
                    // TODO: implement function stuff?
                    return Ok(g.add_node(GeneralizedExpr::DumbNode("funkkk".to_owned())));
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
