from itertools import combinations
from math import comb, factorial
from typing import Callable, Optional

import hypernetx as hnx
import torch
import torch_geometric
from hypernetx.reports import edge_size_dist
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.hypergraph2simplicial.base import (
    Hypergraph2SimplicialLifting,
)


class WeightedSimplicialLifting(Hypergraph2SimplicialLifting):
    def __init__(
        self,
        bare_weight_generator: Optional[
            Callable[[SimplicialComplex, hnx.Hypergraph, dict], str]
        ] = None,
        complex_dim=-1,
        **kwargs,
    ):
        """Initialize a WeightedSimplicialLifting

        Args:
            bare_weight_generator (Optional[ Callable[[SimplicialComplex, hnx.Hypergraph], str] ], optional): A function which takes the hypergraph, associated simplicial complex, and hyperedge attributes dict, and attaches 'bare affinity weights' to a subset of the simplices. It should return the name of the attribute which contains the bare affinity weights. If None, defaults to the collaboration_network_affinity_weights function.
            complex_dim (int, optional): Dimension of the simplicial complex. Defaults to -1, which matches dimension of the hypergraph.
        """
        super().__init__(complex_dim, **kwargs)
        if bare_weight_generator:
            self.bare_weight_generator = bare_weight_generator
        else:
            self.bare_weight_generator = collaboration_network_affinity_weights

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """Lifts the topology of a hypergraph to weighted simplicial complex. Based on the paper [1].

        [[1]](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.034319) Baccini, Federica, Filippo Geraci, and Ginestra Bianconi. "Weighted simplicial complexes and their representation power of higher-order network data and topology." Physical Review E 106.3 (2022): 034319.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        H, edge_attrs = self._generate_hypergraph_from_data(data)

        # Generate the simplicial complex by 'filling in' the edges of H with all subsets
        simplicial_closure = SimplicialComplex.simplicial_closure_of_hypergraph(H)

        # If necessary, reduce the dimension of the closure to self.complex_dim
        if self.complex_dim not in [-1, simplicial_closure.dim]:
            dim = min(simplicial_closure.dim, self.complex_dim)
            simplicial_closure = simplicial_closure.restrict_to_simplices(
                simplicial_closure.skeleton(dim)
            )

        # Generate the weights
        bare_weight_key = self.bare_weight_generator(simplicial_closure, H, edge_attrs)
        _bare_affinity_weights_to_topological_weights(
            simplicial_closure, bare_weight_key=bare_weight_key
        )

        lifted_topology = get_complex_connectivity(
            simplicial_closure, simplicial_closure.dim, signed=self.signed
        )
        for rank in range(simplicial_closure.dim):
            lifted_topology[f"topological_weights_{rank}"] = torch.stack(
                list(
                    simplicial_closure.get_simplex_attributes(
                        "topological_weight", rank=rank
                    ).values()
                )
            )

        return lifted_topology


def _bare_affinity_weights_to_topological_weights(
    sc: SimplicialComplex, bare_weight_key: str = "bare_weight"
) -> None:
    """Take 'bare affinity weights' (>=0) associated with each simplex and transform to 'topological weights' (> 0)."""
    bare_weights = sc.get_simplex_attributes(bare_weight_key)
    if not bare_weights:
        raise ValueError(
            f"No bare weights found for key {bare_weight_key}. You must set bare weights on a non-empty subset of simplices."
        )

    sc.set_simplex_attributes(bare_weights, "topological_weight")
    for rank in range(sc.dim, 0, -1):
        for simplex in sc.skeleton(rank):
            for b in sc.get_boundaries([simplex], min_dim=rank - 1, max_dim=rank - 1):
                sc[b]["topological_weight"] = sc[simplex].get(
                    "topological_weight", 0
                ) + sc[b].get(bare_weight_key, 0)
    return


def collaboration_network_affinity_weights(
    sc: SimplicialComplex,
    H: hnx.Hypergraph,
    edge_attrs: dict,
) -> str:
    """Generate bare affinity weights for a simplicial complex based on a collaboration hypergraph H. These weights are designed to have the following property: if the edges in the hypergraph correspond to author groups of papers, the topological weight of a node will be the number of papers that author has written.

    Args:
        sc (SimplicialComplex): The simplicial complex lifting of H.
        H (hnx.Hypergraph): Hypergraph from which sc is generated.
    """
    # Edge attribute to which bare affinity weights are assigned
    bare_weight_key: str = "bare_weight"

    d = sc.dim
    if d == max(edge_size_dist(H)) - 1:
        for edge_index in H.edges:
            edge = H.edges[edge_index]
            sc[edge][bare_weight_key] = edge_attrs.get(edge_index, 1) / factorial(
                len(edge) - 1
            )
        return bare_weight_key

    for edge_index in H.edges:
        edge = H.edges[edge_index]
        if len(edge) <= d:
            sc[edge][bare_weight_key] = edge_attrs.get(edge_index, 1) / factorial(
                len(edge) - 1
            )
        else:
            n_prime = len(edge) - 1
            edge_factor = edge_attrs.get(edge_index, 1) / (
                factorial(d) * comb(n_prime, d)
            )
            for simplex in combinations(edge, d):
                sc[simplex][bare_weight_key] = (
                    sc[simplex].get(bare_weight_key, 0) + edge_factor
                )

    return bare_weight_key
