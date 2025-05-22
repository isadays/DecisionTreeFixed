# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch sklearn.tree._tree.Tree.__setstate__ to auto-upgrade old pickles
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import sklearn.tree._tree as _tree_mod
import joblib

# Keep a reference to the original
_orig_setstate = _tree_mod.Tree.__setstate__

def _patched_setstate(self, state):
    nodes = state.get("nodes")
    # if it's a legacy 7-field array, inject the 8th field:
    if nodes is not None and "missing_go_to_left" not in nodes.dtype.names:
        descr = nodes.dtype.descr + [("missing_go_to_left", "u1")]
        new_dtype = np.dtype(descr, align=False)
        new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
        # copy old fields
        for name in nodes.dtype.names:
            new_nodes[name] = nodes[name]
        # fill the new field however you like (0 sends missing→right)
        new_nodes["missing_go_to_left"] = 0
        state["nodes"] = new_nodes
    # now call the real __setstate__
    return _orig_setstate(self, state)


# apply the patch
_tree_mod.Tree.__setstate__ = _patched_setstate

# ─────────────────────────────────────────────────────────────────────────────
# Now it’s safe to load your pickle:


model = joblib.load("model.pkl")
print("✅ loaded under sklearn", model)  
