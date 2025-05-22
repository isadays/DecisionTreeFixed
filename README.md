# DecisionTreeFixed
Solution to the problem File "sklearn/tree/_tree.pyx", line 728, in sklearn.tree._tree.Tree.__setstate__   File "sklearn/tree/_tree.pyx", line 1434, in sklearn.tree._tree._check_node_ndarray



### Solution

You can work around this incompatibility entirely inside your notebook by monkey-patching the Tree.__setstate__ method to inject the missing missing_go_to_left field on-the-fly before it does its dtype check. Drop this cell before your joblib. load(...) and you’ll never hit that error.
