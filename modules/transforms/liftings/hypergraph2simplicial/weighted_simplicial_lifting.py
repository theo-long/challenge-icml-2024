from modules.transforms.liftings.hypergraph2simplicial.base import (
    Hypergraph2SimplicialLifting,
)


class WeightedSimplicialLifting(Hypergraph2SimplicialLifting):
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        """Lifts the topology of a hypergraph to weighted simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        return super().lift_topology(data)
