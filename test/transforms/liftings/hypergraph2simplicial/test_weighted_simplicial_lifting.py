"""Test the message passing module."""

import torch

from modules.data.utils.utils import load_manual_graph
from modules.transforms.liftings.graph2simplicial.clique_lifting import (
    WeightedSimplicialLifting,
)


class TestWeightedSimplicialLifting:
    """Test the WeightedSimplicialLifting class."""

    def setup_method(self):
        # Load hypergraph dataset for testing - TODO
        self.data = None

        # Initialise the WeightedSimplicialLiftingLifting class
        self.lifting = WeightedSimplicialLifting(complex_dim=3)

    def test_lift_topology(self):
        """Test the lift_topology method."""
        raise NotImplementedError
