import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from modules.transforms.liftings.hypergraph2simplicial.base import (
    Hypergraph2SimplicialLifting,
)


class WeightedSimplicialLifting(Hypergraph2SimplicialLifting):
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
        hyperedges = data.incidence_hyperedges
        bare_affinity_weights = torch.zeros(data.num_hyperedges)


def _bare_affinity_weights_to_topological_weights(
    sc: SimplicialComplex,
):
    """Take 'bare affinity weights' (>=0) associated with each simplex and transform to 'topological weights' (> 0)."""
    sc.set_simplex_attributes(
        sc.get_simplex_attributes("bare_weight"), "topological_weight"
    )
    for rank in range(sc.dim, 0, -1):
        for simplex in sc.skeleton(rank):
            for b in sc.get_boundaries([simplex], min_dim=rank - 1, max_dim=rank - 1):
                sc[b]["topological_weight"] += sc[simplex]["bare_weight"]
    return sc


def _generate_bare_affinity_weights(sc: SimplicialComplex):
    pass
