use super::{ArgType, CustomFunc, LispExpr, LispFunc, LispMacro, LispValue};
use std::collections::HashMap;
use std::ops::Index;
use petgraph::Undirected;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::{Config, Dot};
use std::fmt;

#[derive(Debug)]
pub enum SpecializedType {
    Boolean,
    Integer,
}

#[derive(Clone)]
struct FunctionReference {
    ins: Vec<NodeIndex>,
    out: NodeIndex,
}

#[derive(Debug)]
pub enum GeneralizedExpr {
    Expr(LispExpr),
    FixedType(SpecializedType),
    DumbNode(String),
}

impl fmt::Display for GeneralizedExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GeneralizedExpr::Expr(ref e) => e.fmt(f),
            GeneralizedExpr::FixedType(ref e) => write!(f, "{:?}", e),
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


// TODO: we should actually take the arguments instead of their types
//       for when a function is passed in.
pub fn make_specialization_graph(
    f: CustomFunc,
    argument_types: &[ArgType],
) -> Result<Graph<GeneralizedExpr, NoLabel, Undirected>, ()> {
    let mut g = Graph::<GeneralizedExpr, NoLabel, Undirected>::new_undirected();
    let mut map = HashMap::new();
    let bool_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Boolean));
    let int_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Integer));

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
        argument_types,
        &mut g,
        &mut map,
        bool_idx,
        int_idx,
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

fn expand_graph(
    expr: &LispExpr,
    argument_types: &[ArgType],
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    reference_map: &mut HashMap<LispFunc, FunctionReference>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
) -> Result<NodeIndex, ()> {
    let self_node = g.add_node(GeneralizedExpr::Expr(expr.clone()));

    match *expr {
        LispExpr::Argument(index, _c) => match argument_types[index] {
            ArgType::Boolean => {
                g.add_edge(self_node, bool_index, NoLabel);
            }
            ArgType::Integer => {
                g.add_edge(self_node, int_index, NoLabel);
            }
            ArgType::Function => {
                // TODO: implement function stuff
                return Ok(g.add_node(
                    GeneralizedExpr::DumbNode("unimplemented".to_owned()),
                ));
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
                                g.add_edge(test_idx, bool_index, NoLabel);

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

                                g.add_edge(branch_a_idx, branch_b_idx, NoLabel);
                                g.add_edge(self_node, branch_a_idx, NoLabel);
                            }
                            // Do not bother with definitions/ lambdas (for now)
                            LispMacro::Define | LispMacro::Lambda => return Err(()),
                        }
                    }
                    _ => {
                        return Ok(g.add_node(
                            GeneralizedExpr::DumbNode("unimplemented".to_owned()),
                        ))
                    }
                }
            } else {
                panic!("empty call shouldnt be possible!");
            }
        }
        LispExpr::Value(ref val) => {
            match *val {
                LispValue::Boolean(..) => {
                    // TODO: dedup with argument type
                    g.add_edge(self_node, bool_index, NoLabel);
                }
                LispValue::Integer(..) => {
                    g.add_edge(self_node, int_index, NoLabel);
                }
                LispValue::Function(..) => {
                    // TODO: implement function stuff
                    return Ok(g.add_node(
                        GeneralizedExpr::DumbNode("unimplemented".to_owned()),
                    ));
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
