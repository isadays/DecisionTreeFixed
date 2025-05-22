import sklearn.tree._tree as _tm
_orig_Tree = _tm.Tree

class PatchedTree(_orig_Tree):
    def __setstate__(self, state):
        nodes = state.get("nodes")
        if nodes is not None and "missing_go_to_left" not in nodes.dtype.names:
            descr = nodes.dtype.descr + [("missing_go_to_left", "u1")]
            dt    = np.dtype(descr, align=False)
            new   = np.zeros(nodes.shape, dtype=dt)
            for f in nodes.dtype.names: 
                new[f] = nodes[f]
            new["missing_go_to_left"] = 0
            state["nodes"] = new
        super(PatchedTree, self).__setstate__(state)

_tm.Tree = PatchedTree
print("Applied Tree subclass for legacy unpickling")

clf_old = joblib.load(MODEL_PATH)
print("Leaf indices under <1.3:", clf_old.apply(X[:5].astype(float)))
