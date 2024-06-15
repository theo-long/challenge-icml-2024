from itertools import combinations
from math import comb, factorial

import hypernetx as hnx
import torch_geometric
from hypernetx.reports import edge_size_dist
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.hypergraph2simplicial.base import (
    Hypergraph2SimplicialLifting,
)


class WeightedSimplicialLifting(Hypergraph2SimplicialLifting):
    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(complex_dim, **kwargs)

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
        H = self._generate_hypergraph_from_data(data)

        # Generate the simplicial complex by 'filling in' the edges of H with all subsets
        simplicial_closure = SimplicialComplex.simplicial_closure_of_hypergraph(H)

        # If necessary, reduce the dimension of the closure to self.complex_dim
        if self.complex_dim not in [-1, simplicial_closure.dim]:
            dim = min(simplicial_closure.dim, self.complex_dim)
            simplicial_closure = simplicial_closure.restrict_to_simplices(
                simplicial_closure.skeleton(dim)
            )

        # Generate the weights
        _generate_bare_affinity_weights(simplicial_closure, H)
        _bare_affinity_weights_to_topological_weights(simplicial_closure)

        return simplicial_closure


def _bare_affinity_weights_to_topological_weights(
    sc: SimplicialComplex, bare_weight_key="bare_weight"
):
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
                sc[b]["topological_weight"] += sc[simplex][bare_weight_key]
    return


def _generate_bare_affinity_weights(
    sc: SimplicialComplex,
    H: hnx.Hypergraph,
    bare_weight_key: str = "bare_weight",
):
    d = sc.dim
    if d == max(edge_size_dist(H)):
        for edge in H.edges:
            sc[edge][bare_weight_key] = H[edge].get("edge_attr", 1) / factorial(
                len(H[edge]) - 1
            )
        return

    for edge in H.edges:
        if len(edge) <= d:
            sc[edge][bare_weight_key] = H[edge].get("edge_attr", 1) / factorial(
                len(H[edge]) - 1
            )
        else:
            n_prime = len(H[edge]) - 1
            edge_factor = H[edge].get("edge_attr", 1) / (
                factorial(d) * comb(n_prime, d)
            )
            for simplex in combinations(edge, d):
                sc[simplex][bare_weight_key] = (
                    sc[simplex].get(bare_weight_key, 0) + edge_factor
                )
