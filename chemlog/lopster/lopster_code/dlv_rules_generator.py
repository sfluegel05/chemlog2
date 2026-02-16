from typing import List, Set
from .atom import Atom
from .dlv_rule import DLVRule
from .description_graph import DescriptionGraph
from .node import Node

class DLVRulesGenerator:
    class GraphAtomMode:
        FUNCTION_TERMS_ON = 1
        VARIABLES_ON = 2
        CONSTANTS_ON = 3

    def translate_dg_into_rules(self, description_graph: DescriptionGraph) -> Set[DLVRule]:
        rules = set()
        rules.update(self.produce_layout_rules(description_graph, for_msa_check=False))
        rules.add(self.produce_start_concept_fact(description_graph))
        return rules

    def produce_start_rule(self, description_graph: DescriptionGraph) -> DLVRule:
        head = self.produce_graph_atom(description_graph, self.GraphAtomMode.FUNCTION_TERMS_ON)
        body = [self.produce_start_atom(description_graph, "X")]
        return DLVRule(head, body)

    def produce_msa_start_rule(self, description_graph: DescriptionGraph) -> DLVRule:
        head = self.produce_graph_atom(description_graph, self.GraphAtomMode.CONSTANTS_ON)
        body = [self.produce_start_atom(description_graph, "X")]
        return DLVRule(head, body)

    def produce_layout_rules(self, description_graph: DescriptionGraph, for_msa_check: bool) -> Set[DLVRule]:
        rules = set()
        body_atoms = [self.produce_start_atom(description_graph.get_start_concept())]

        for node in description_graph.get_nodes():
            for unary_pred in node.get_label():
                term = (self.build_constant_layout_term(description_graph.get_start_concept(), node.get_number())
                        if for_msa_check else
                        self.build_function_layout_term(description_graph.get_start_concept(), node.get_number()))
                head = Atom(unary_pred, [term])
                rules.add(DLVRule(head, body_atoms))

        for edge in description_graph.get_edges():
            for binary_pred in edge.get_label():
                from_term = (self.build_constant_layout_term(description_graph.get_start_concept(), edge.get_from_node())
                             if for_msa_check else
                             self.build_function_layout_term(description_graph.get_start_concept(), edge.get_from_node()))
                to_term = (self.build_constant_layout_term(description_graph.get_start_concept(), edge.get_to_node())
                           if for_msa_check else
                           self.build_function_layout_term(description_graph.get_start_concept(), edge.get_to_node()))
                head = Atom(binary_pred, [from_term, to_term])
                rules.add(DLVRule(head, body_atoms))

        return rules

    def produce_start_atom(self, start_concept_or_graph, variable: str = None) -> Atom:
        if isinstance(start_concept_or_graph, DescriptionGraph):
            sc = start_concept_or_graph.get_start_concept()
        else:
            sc = start_concept_or_graph
        var = variable or "X"
        return Atom(sc, [var])

    def build_function_layout_term(self, start_concept: str, node_number: int) -> str:
        if node_number != 0:
            return f"f_{start_concept}_{node_number}(X)"
        return "X"

    def build_constant_layout_term(self, start_concept: str, node_number: int) -> str:
        if node_number != 0:
            return f"c_{start_concept}_{node_number}"
        return "X"

    def produce_graph_atom(self, description_graph: DescriptionGraph, graph_atom_mode: int) -> Atom:
        pred = f"g_{description_graph.get_start_concept()}"
        terms = []
        if graph_atom_mode in (self.GraphAtomMode.FUNCTION_TERMS_ON, self.GraphAtomMode.CONSTANTS_ON):
            terms.append("X")
        elif graph_atom_mode == self.GraphAtomMode.VARIABLES_ON:
            terms.append("Y0")

        node_count = len(description_graph.get_nodes())
        # nodes are expected to be numbered 0..n-1 but set iteration is unordered; we mimic original by using range
        for i in range(1, node_count):
            if graph_atom_mode == self.GraphAtomMode.FUNCTION_TERMS_ON:
                terms.append(f"f_{description_graph.get_start_concept()}_{i}(X)")
            elif graph_atom_mode == self.GraphAtomMode.VARIABLES_ON:
                terms.append(f"Y{i}")
            elif graph_atom_mode == self.GraphAtomMode.CONSTANTS_ON:
                terms.append(f"c_{description_graph.get_start_concept()}_{i}")
        return Atom(pred, terms)

    def produce_start_concept_fact(self, description_graph: DescriptionGraph) -> DLVRule:
        head = Atom(description_graph.get_start_concept(), [f"c_{description_graph.get_start_concept()}"])
        return DLVRule(head, [])

    # Additional MSA related methods (succession, tag, cycle detection, trans closure) can be added similarly
