from typing import Set, Dict
from .description_graph import DescriptionGraph
from .node import Node
from .edge import Edge
from rdkit import Chem


class DGGenerator:
    """Builds description graph objects from molfiles using RDKit."""
    
    def assemble_dg_map(self, mol_list) -> Dict[str, DescriptionGraph]:
        """
        Produces a map of strings and description graph objects by processing 
        the molfiles located in input_files_path. The size of the map is specified 
        by size_of_dg_set. Discards molecules that have an R atom (radicals).
        """
        description_graphs = {}
        
        # Convert mol files into DescriptionGraph objects
        mol_index = 0
        while mol_index < len(mol_list):
            try:
                dg = self.build_description_graph(mol_list[mol_index], f"mol_{mol_index}")
                if self.is_radical_free(dg):
                    description_graphs[f"mol_{mol_index}"] = dg
            except Exception as e:
                print(f"Error processing {mol_index}: {e}")
            mol_index += 1
        
        return description_graphs

    def build_description_graph(self, mol: Chem.Mol, mol_index: str) -> DescriptionGraph:
        """
        Given a molfile path, produces a DescriptionGraph object.
        """        
        if mol is None:
            raise ValueError(f"Could not parse mol: {mol_index}")
        
        nodes = self.retrieve_nodes(mol, mol_index)
        edges = self.retrieve_edges(mol, mol_index)
        
        return DescriptionGraph(nodes, edges, mol_index)

    def is_radical_free(self, dg: DescriptionGraph) -> bool:
        """Detects whether the graph contains a node labeled with 'r' (radical)."""
        for node in dg.get_nodes():
            if 'r' in node.get_label():
                return False
        return True

    def retrieve_nodes(self, mol: Chem.Mol, molecule_name: str) -> Set[Node]:
        """Returns a set of Node objects from an RDKit molecule."""
        nodes = set()
        
        # Node 0: root molecule node
        label_zero_node = {"molecule"}
        nodes.add(Node(0, label_zero_node))
        
        # Nodes 1..n: one per atom
        number_of_atoms = mol.GetNumAtoms()
        for i in range(number_of_atoms):
            node_index = i + 1
            atom = mol.GetAtomWithIdx(i)
            label = set()
            
            # Add element symbol (lowercase) - exclude wildcards
            if atom.GetAtomicNum() > 0:
                label.add(atom.GetSymbol().lower())
            
            # Add formal charge labels
            charge = atom.GetFormalCharge()
            if charge == 1:
                label.add("plus1")
            elif charge == 2:
                label.add("plus2")
            elif charge == 3:
                label.add("plus3")
            elif charge == -1:
                label.add("minus1")
            elif charge == -2:
                label.add("minus2")
            elif charge == -3:
                label.add("minus3")
            
            nodes.add(Node(node_index, label))
        
        return nodes

    def retrieve_edges(self, mol: Chem.Mol, molecule_name: str) -> Set[Edge]:
        """Returns a set of Edge objects from an RDKit molecule."""
        edges = set()
        
        # Map atoms to node indices (1-based, 0 is root)
        number_of_atoms = mol.GetNumAtoms()
        root_node = 0
        
        # Add "hasAtom" edges from root to all atoms
        label_has_atom = {"hasAtom"}
        for i in range(number_of_atoms):
            node_index = i + 1
            edge = Edge(root_node, node_index, label_has_atom)
            edges.add(edge)
        
        # Add bond edges (bidirectional)
        number_of_bonds = mol.GetNumBonds()
        for i in range(number_of_bonds):
            bond = mol.GetBondWithIdx(i)
            from_node = bond.GetBeginAtomIdx() + 1
            to_node = bond.GetEndAtomIdx() + 1
            
            # Bond order as lowercase string
            bond_type = bond.GetBondType()
            bond_order = str(bond_type).lower()
            
            label = {bond_order}
            edge_forward = Edge(from_node, to_node, label)
            edge_backward = Edge(to_node, from_node, label)
            edges.add(edge_forward)
            edges.add(edge_backward)
        
        return edges

