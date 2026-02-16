from typing import Set
from .node import Node
from .edge import Edge

class DescriptionGraph:
    """Data structure for description graphs: set of nodes, edges and start concept."""
    def __init__(self, nodes: Set[Node], edges: Set[Edge], start_concept: str):
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.start_concept = start_concept

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_start_concept(self):
        return self.start_concept

    def __repr__(self) -> str:
        parts = ["[", "Nodes", "--------------------"]
        for n in self.nodes:
            label = ','.join(n.get_label())
            parts.append(f"   {n.get_number()} : {label}.")
        parts.append("\nEdges")
        parts.append("--------------------")
        for e in self.edges:
            label = ','.join(e.get_label())
            parts.append(f"  {e.get_from_node()} --> {e.get_to_node()} : {label}.")
        parts.append("\nStart Concept")
        parts.append("--------------------")
        parts.append(self.start_concept)
        parts.append("]")
        return "\n".join(parts)
