use super::{ArgType, BuiltIn, CustomFunc, LispExpr, LispFunc, LispMacro, LispValue};
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

fn create_ref_map(
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
) -> HashMap<LispFunc, FunctionReference> {
    let mut map = HashMap::new();

    // TODO: find a way to make sure we do not forget a built-in?
    map.insert(LispFunc::BuiltIn(BuiltIn::AddOne), {
        let in_node = g.add_node(GeneralizedExpr::DumbNode("add1 in".to_owned()));
        g.add_edge(in_node, int_index, NoLabel);
        let out_node = g.add_node(GeneralizedExpr::DumbNode("add1 out".to_owned()));
        g.add_edge(out_node, int_index, NoLabel);
        FunctionReference {
            ins: vec![in_node],
            out: out_node,
        }
    });

    map.insert(LispFunc::BuiltIn(BuiltIn::SubOne), {
        let in_node = g.add_node(GeneralizedExpr::DumbNode("sub1 in".to_owned()));
        g.add_edge(in_node, int_index, NoLabel);
        let out_node = g.add_node(GeneralizedExpr::DumbNode("sub1 out".to_owned()));
        g.add_edge(out_node, int_index, NoLabel);
        FunctionReference {
            ins: vec![in_node],
            out: out_node,
        }
    });

    map.insert(LispFunc::BuiltIn(BuiltIn::CheckZero), {
        let in_node = g.add_node(GeneralizedExpr::DumbNode("zero? in".to_owned()));
        g.add_edge(in_node, int_index, NoLabel);
        let out_node = g.add_node(GeneralizedExpr::DumbNode("zero? out".to_owned()));
        g.add_edge(out_node, bool_index, NoLabel);
        FunctionReference {
            ins: vec![in_node],
            out: out_node,
        }
    });

    map
}


// TODO: we should actually take the arguments instead of their types
//       for when a function is passed in.
pub fn make_specialization_graph(
    f: CustomFunc,
    argument_types: &[ArgType],
) -> Result<Graph<GeneralizedExpr, NoLabel, Undirected>, ()> {
    let mut g = Graph::<GeneralizedExpr, NoLabel, Undirected>::new_undirected();
    let bool_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Boolean));
    let int_idx = g.add_node(GeneralizedExpr::FixedType(SpecializedType::Integer));
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

    // TODO: we should add edges between in nodes and types here?
    // and then pass the list of node indices of the ins around

    let res = expand_graph(&(*f.body), &main_ref, &mut g, &mut map, bool_idx, int_idx)?;

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
    main_ref: &FunctionReference,
    g: &mut Graph<GeneralizedExpr, NoLabel, Undirected>,
    reference_map: &mut HashMap<LispFunc, FunctionReference>,
    bool_index: NodeIndex,
    int_index: NodeIndex,
) -> Result<NodeIndex, ()> {
    match *expr {
        LispExpr::Argument(index, _c) => Ok(main_ref.ins[index]),
        LispExpr::Call(ref arg_vec, _is_tail_call, is_self_call) => {
            let self_node = g.add_node(GeneralizedExpr::Expr(expr.clone()));

            if let [ref head, ref tail..] = arg_vec[..] {
                match *head {
                    _ if is_self_call => {
                        let arg_indices = tail.iter()
                            .map(|arg_expr| {
                                expand_graph(
                                    arg_expr,
                                    main_ref,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        // Recursion
                        assert_eq!(arg_indices.len(), main_ref.ins.len(), "bad recursion");

                        for (&top_node, &arg_node) in main_ref.ins.iter().zip(arg_indices.iter()) {
                            g.add_edge(top_node, arg_node, NoLabel);
                        }

                        g.add_edge(self_node, main_ref.out, NoLabel);
                    }
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
                                    main_ref,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;
                                g.add_edge(test_idx, bool_index, NoLabel);

                                let branch_a_idx = expand_graph(
                                    tail.index(1),
                                    main_ref,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;
                                let branch_b_idx = expand_graph(
                                    tail.index(2),
                                    main_ref,
                                    g,
                                    reference_map,
                                    bool_index,
                                    int_index,
                                )?;

                                g.add_edge(self_node, branch_b_idx, NoLabel);
                                g.add_edge(self_node, branch_a_idx, NoLabel);
                            }
                            // Do not bother with definitions/ lambdas (for now)
                            LispMacro::Define | LispMacro::Lambda => return Err(()),
                        }
                    }
                    LispExpr::Value(ref val) => match *val {
                        LispValue::Function(ref f) => {
                            let arg_indices = tail.iter()
                                .map(|arg_expr| {
                                    expand_graph(
                                        arg_expr,
                                        main_ref,
                                        g,
                                        reference_map,
                                        bool_index,
                                        int_index,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            if let Some(reference) = reference_map.get(f) {
                                if reference.ins.len() != tail.len() {
                                    return Err(());
                                }

                                for (&idx, &function_in) in
                                    arg_indices.iter().zip(reference.ins.iter())
                                {
                                    g.add_edge(idx, function_in, NoLabel);
                                }

                                println!("{}", expr.clone());

                                g.add_edge(self_node, reference.out, NoLabel);
                            } else {
                                return Ok(g.add_node(
                                    GeneralizedExpr::DumbNode("unknown function".to_owned()),
                                ));
                            }
                        }
                        _ => {
                            return Ok(g.add_node(
                                GeneralizedExpr::DumbNode("non-function value".to_owned()),
                            ))
                        }
                    },

                    _ => {
                        return Ok(g.add_node(GeneralizedExpr::DumbNode(
                            "non-value function head?".to_owned(),
                        )))
                    }
                }
            } else {
                panic!("empty call shouldnt be possible!");
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
                LispValue::Function(..) => {
                    // TODO: implement function stuff
                    return Ok(g.add_node(GeneralizedExpr::DumbNode("funkkk".to_owned())));
                }
                LispValue::List(..) => return Err(()),
            }

            Ok(self_node)
        }
        // All references should be resolved at this point, right?
        LispExpr::OpVar(..) => unreachable!(),
        // We shouldn't find a top level macro - this is almost surely a faulty
        // expression.
        LispExpr::Macro(_) => return Err(()),
    }
}
