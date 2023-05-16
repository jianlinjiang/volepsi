use core::panic;
use std::cell::RefCell;
use std::collections::HashSet;
#[derive(Debug, Clone)]
pub struct WeightNode {
    pub weight: usize,
    pub prev_node: usize,
    pub next_node: usize,
}

#[derive(Debug, Clone)]
pub struct WeightData {
    pub nodes: Vec<WeightNode>,
    pub weight_sets: Vec<Option<*mut WeightNode>>,
}

static NULL_NODE: usize = usize::MAX;

fn idx_of(node: *const WeightNode, start: *const WeightNode) -> usize {
    unsafe { (node as *const WeightNode).offset_from(start as *const WeightNode) as usize }
}

impl<'a> WeightData {
    pub fn idx_of(&self, node: &WeightNode) -> usize {
        unsafe {
            (node as *const WeightNode).offset_from(&self.nodes[0] as *const WeightNode) as usize
        }
    }

    pub fn validate(&mut self) {
        let mut nodes: HashSet<usize> = HashSet::new();
        for i in 0..self.weight_sets.len() {
            let mut head = self.weight_sets[i];
            while head.is_some() {
                let pointer = head.unwrap();
                unsafe {
                    assert_eq!((*pointer).weight, i as usize);
                    assert_eq!(nodes.insert(self.idx_of(&(*pointer)) as usize), true);
                    if (*pointer).next_node != NULL_NODE {
                        assert_eq!(
                            self.nodes[(*pointer).next_node as usize].prev_node,
                            self.idx_of(&(*pointer)) as usize
                        );
                        head = Some(&mut self.nodes[(*pointer).next_node as usize]);
                    } else {
                        head = None;
                    }
                }
            }
        }
    }

    pub fn init(weights: &[usize]) -> WeightData {
        let mut nodes: Vec<WeightNode> = vec![
            WeightNode {
                weight: 0,
                prev_node: NULL_NODE,
                next_node: NULL_NODE
            };
            weights.len()
        ];

        let mut weight_sets: Vec<Option<*mut WeightNode>> = vec![None; 200];
        let start: *const WeightNode = &nodes[0];

        weights.iter().enumerate().for_each(|(idx, &weight)| {
            if weight >= weight_sets.len() {
                panic!("something went wrong, maybe duplicate inputs.");
            }
            let mut node = &mut nodes[idx];
            node.weight = weight;
            node.prev_node = NULL_NODE;
            node.next_node = NULL_NODE;

            let ws = &weight_sets[weight as usize];
            if ws.is_some() {
                let w: *mut WeightNode = ws.unwrap();
                unsafe {
                    assert_eq!((*w).prev_node, NULL_NODE);
                    (*w).prev_node = idx_of(node, start) as usize;
                    node.next_node = idx_of(w, start) as usize;
                }
            }
            weight_sets[weight] = Some(node as *mut WeightNode);
        });
        while weight_sets.last().is_none() {
            weight_sets.pop();
        }


        // for i in 0..weights.len() {
        //     let mut node: &mut WeightNode = &mut nodes[i];
        //     node.weight = weights[i];
        //     node.prev_node = NULL_NODE;
        //     node.next_node = NULL_NODE;

        //     if node.weight as usize >= weight_sets.len() {
        //         panic!("something went wrong, maybe duplicate inputs.");
        //     }

        //     let ws = &weight_sets[node.weight as usize];

        //     if ws.is_some() {
        //         let w: *mut WeightNode = ws.unwrap();
        //         unsafe {
        //             assert_eq!((*w).prev_node, NULL_NODE);
        //             (*w).prev_node = idx_of(node, start) as usize;
        //             node.next_node = idx_of(w, start) as usize;
        //         }
        //     }
        //     weight_sets[node.weight as usize] = Some(node as *mut WeightNode);
        // }
        // let mut i = weight_sets.len() - 1;
        // loop {
        //     if weight_sets[i].is_some() {
        //         weight_sets.resize(i + 1, None);
        //         break;
        //     }
        //     i = i - 1;
        // }
        WeightData { nodes, weight_sets }
    }

    pub fn get_min_weightnode(&self) -> *mut WeightNode {
        for i in 1..self.weight_sets.len() {
            if self.weight_sets[i].is_some() {
                return self.weight_sets[i].unwrap();
            }
        }
        panic!("can't find min weight node");
    }

    pub fn push_node(&mut self, node: *mut WeightNode) {
        unsafe {
            let weight: usize = (*node).weight as usize;
            assert_eq!((*node).next_node, NULL_NODE);
            assert_eq!((*node).prev_node, NULL_NODE);

            if self.weight_sets.len() <= weight as usize {
                self.weight_sets.resize(weight + 1, None);
            }

            if self.weight_sets[weight].is_none() {
                self.weight_sets[weight] = Some(node);
            } else {
                let ptr = self.weight_sets[weight].unwrap();
                assert_eq!((*ptr).prev_node, NULL_NODE);
                (*ptr).prev_node = self.idx_of(&(*node)) as usize;
                (*node).next_node = self.idx_of(&(*ptr)) as usize;
                self.weight_sets[weight] = Some(node);
            }
        }
    }

    pub fn pop_node(&mut self, node: *mut WeightNode) {
        unsafe {
            let weight: usize = (*node).weight as usize;
            if (*node).prev_node == NULL_NODE {
                // 链表头部
                assert_eq!(self.weight_sets[weight].unwrap(), node);

                if (*node).next_node == NULL_NODE {
                    // 链表里仅有这一个元素
                    self.weight_sets[weight] = None;
                    while self.weight_sets.last().unwrap().is_none() {
                        self.weight_sets.pop();
                    }
                } else {
                    self.weight_sets[weight] =
                        Some(&mut self.nodes[(*node).next_node as usize] as *mut WeightNode);
                    (*self.weight_sets[weight].unwrap()).prev_node = NULL_NODE;
                }
            } else {
                // 在链表中间
                let start = &mut self.nodes[0] as *mut WeightNode;
                {
                    let prev: *mut WeightNode = start.offset((*node).prev_node as isize);
                    if (*node).next_node == NULL_NODE {
                        (*prev).next_node = NULL_NODE;
                    } else {
                        let next: *mut WeightNode = start.offset((*node).next_node as isize);
                        (*prev).next_node = (*node).next_node;
                        (*next).prev_node = (*node).prev_node;
                    }
                }
            }
            (*node).prev_node = NULL_NODE;
            (*node).next_node = NULL_NODE;
        }
    }

    pub fn decement_weight(&mut self, node: *mut WeightNode) {
        unsafe {
            assert!((*node).weight != 0);
            self.pop_node(node);
            (*node).weight -= 1;
            self.push_node(node);
        }
    }

    pub fn decrease_column_weight(&mut self, col_idx: usize) {
        let node = &mut self.nodes[col_idx] as *mut WeightNode;
        self.pop_node(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, RngCore};
    #[test]
    fn offset_test() {
        let nodes: Vec<WeightNode> = vec![
            WeightNode {
                weight: 1,
                prev_node: NULL_NODE,
                next_node: NULL_NODE
            };
            200
        ];
        let node = &nodes[3];
        let offset = idx_of(node as *const WeightNode, &nodes[0] as *const WeightNode);
        // (node as *const WeightNode).offset_from(&nodes[0] as *const WeightNode) as usize
        assert_eq!(offset, 3);
    }

    #[test]
    fn weightdata_test() {
        let mut rng = thread_rng();

        let weights: Vec<usize> = (0..1000)
            .into_iter()
            .map(|_| (rng.next_u64() % 5) as usize)
            .collect();
        let mut data = WeightData::init(&weights);
        println!("weight sets len {}", data.weight_sets.len());
        data.validate();
        let mut i = 0;
        loop {
            if data.nodes[i].weight != 0 {
                unsafe {
                    let node: *mut WeightNode = &mut data.nodes[i] as *mut WeightNode;
                    data.decement_weight(node);
                }
                break;
            } else {
                i += 1;
            }
        }
    }
}
