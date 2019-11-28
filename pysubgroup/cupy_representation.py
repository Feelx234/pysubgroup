import cupy as cp
import pysubgroup as ps

class CupyBitSet_Conjunction(ps.Conjunction):
    n_instances = 0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
                # empty description ==> return a list of all '1's
        if not self._selectors:
            return cp.full(CupyBitSet_Conjunction.n_instances, True, dtype=bool)
        # non-empty description
        return cp.all([sel.representation for sel in self._selectors], axis=0)

    @property
    def size(self):
        return cp.count_nonzero(self.representation)

    def append_and(self, to_append):
        super().append_and(to_append)
        self.representation = cp.logical_and(self.representation, to_append.representation)

    @property
    def __array_interface__(self):
        return self.representation.__array_interface__






class CupyBitSetRepresentation(ps.RepresentationBase):
    Conjunction = CupyBitSet_Conjunction

    def __init__(self, df):
        self.df = df
        super().__init__(CupyBitSet_Conjunction)

    def patch_selector(self, sel):
        sel.representation = cp.asarray(sel.covers(self.df))

    def patch_classes(self):
        CupyBitSet_Conjunction.n_instances = len(self.df)
        super().patch_classes()