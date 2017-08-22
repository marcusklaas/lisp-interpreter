use super::{ArgType, LispExpr, LispFunc, LispMacro, LispValue};
use std::collections::HashMap;
use std::ops::Index;
use petgraph::Undirected;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::{Config, Dot};

#[derive(Debug)]
pub enum SpecializedType {
    Boolean,
    Integer,
}

struct FunctionReference {
    func: LispFunc,
    ins: Vec<NodeIndex>,
    out: NodeIndex,
}

#[derive(Debug)]
pub enum GeneralizedExpr {
    Expr(LispExpr),
    FixedType(SpecializedType),
}

pub fn make_specialization_graph(
    expr: LispExpr,
    argument_types: &[ArgType],
) -> Result<Graph<GeneralizedExpr, (), Undirected>, ()> {
    let mut g = Graph::<GeneralizedExpr, (), Undirected>::new_undirected();
    let mut map = HashMap::new();
    let bool_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Boolean));
    let int_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Integer));

    let res = expand_graph(&expr, argument_types, &mut g, &mut map, bool_idx, int_idx);

    println!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]));

    res.map(|_idx| g)
}

fn expand_graph(
    expr: &LispExpr,
    argument_types: &[ArgType],
    g: &mut Graph<GeneralizedExpr, (), Undirected>,
    reference_map: &mut HashMap<String, FunctionReference>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
) -> Result<NodeIndex, ()> {
    let self_node = g.add_node(GeneralizedExpr::Expr(expr.clone()));

    match *expr {
        LispExpr::Argument(index, _c) => match argument_types[index] {
            ArgType::Boolean => {
                g.add_edge(self_node, bool_index, ());
            }
            ArgType::Integer => {
                g.add_edge(self_node, int_index, ());
            }
            ArgType::Function => {
                // TODO: implement function stuff
                return Err(());
            }
            ArgType::List => return Err(()),
        },
        LispExpr::Call(ref arg_vec, _tail_call, _self_call) => {
            if let [ref head, ref tail..] = arg_vec[..] {
                match *head {
                    LispExpr::Macro(m) => {
                        match m {
                            LispMacro::Cond => {
                                // Do some special thing for conditionals:
                                // connect the test to the bool node
                                // connect the two branches together
                                if tail.len() != 3 {
                                    return Err(());
                                }

                                let test_idx = expand_graph(
                                    tail.index(0),
                                    argument_types,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;
                                g.add_edge(test_idx, bool_index, ());

                                let branch_a_idx = expand_graph(
                                    tail.index(1),
                                    argument_types,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;
                                let branch_b_idx = expand_graph(
                                    tail.index(2),
                                    argument_types,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;

                                g.add_edge(branch_a_idx, branch_b_idx, ());
                                g.add_edge(self_node, branch_a_idx, ());
                            }
                            // Do not bother with definitions/ lambdas (for now)
                            LispMacro::Define | LispMacro::Lambda => return Err(()),
                        }
                    }
                    _ => return Err(()),
                }
            } else {
                panic!("empty call shouldnt be possible!");
            }
        }
        LispExpr::Value(ref val) => {
            match *val {
                LispValue::Boolean(..) => {
                    // TODO: dedup with argument type
                    g.add_edge(self_node, bool_index, ());
                }
                LispValue::Integer(..) => {
                    g.add_edge(self_node, int_index, ());
                }
                LispValue::Function(..) => {
                    // TODO: implement function stuff
                    return Err(());
                }
                LispValue::List(..) => return Err(()),
            }
        }
        // All references should be resolved at this point, right?
        LispExpr::OpVar(..) => unreachable!(),
        // We shouldn't find a top level macro - this is almost surely a faulty
        // expression.
        LispExpr::Macro(m) => return Err(()),
    }

    Ok(self_node)
}
