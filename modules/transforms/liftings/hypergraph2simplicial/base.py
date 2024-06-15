from modules.transforms.liftings.lifting import HypergraphLifting


class Hypergraph2SimplicialLifting(HypergraphLifting):
    r"""Abstract class for lifting hypergraphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Can be any integer >= -1. Default is -1, which means the dimension will be the same as the size of the largest edge in the hypergraph.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=-1, **kwargs):
        super().__init__(**kwargs)
        if self.complex_dim < -1:
            raise ValueError("complex_dim must be >= -1.")
        self.complex_dim = complex_dim
        self.type = "hypergraph2simplicial"
