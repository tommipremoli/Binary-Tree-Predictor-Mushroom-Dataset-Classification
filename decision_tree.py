import numpy as np
from collections import Counter
import pandas as pd

class Node:
    """
    Represents a node in a decision tree.
    Attributes:
    feature : 
        Index of the feature for splitting (None for leaves).
    threshold : float or int
        Split value (None for leaves).
    left, right : Node
        Child nodes for left and right splits.
    value : int
        Class label if leaf, else None.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature 
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value 

    def is_leaf_node(self):
        return self.value is not None 

class TreePredictor:
    def __init__(self, min_samples_split=2, max_depth=100, criterion="entropy"):
        """
        Builds a decision tree.

        Parameters:
        - min_samples_split: Minimum samples to split.
        - max_depth: Maximum tree depth.
        - criterion: Split criterion ("entropy", "gini", "MSE").

        Methods:
        - fit(X, y): Fits the tree to the data.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth 
        self.root = None
        self.categorical_features = None  
        self.criterion = criterion  

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.categorical_features = X.dtypes == 'object'
            self.categorical_features = self.categorical_features.values
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values
        self.root = self._grow_tree(X, y)

    def compute_zero_one_loss(self, X, y):
        """
        Computes the training error using zero-one loss.
        """
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        total_samples = len(y)
        error_rate = errors / total_samples
        return error_rate
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
        - X: Features.
        - y: Labels.
        - depth: Current tree depth.

        Returns:
        - Node: A decision tree node.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_thresh = self._best_split(X, y)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh, best_feature)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y):
        """
        Finds the best feature and threshold for splitting.

        Parameters:
        - X: Features.
        - y: Labels.

        Returns:
        - split_idx: Index of the best feature.
        - split_threshold: Best threshold for the split.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in range(X.shape[1]):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold, feat_idx)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold, feat_idx):
        """
        Calculates the information gain for a given split.

        Parameters:
        - y: Labels.
        - X_column: Feature values.
        - threshold: Split threshold.
        - feat_idx: Feature index.

        Returns:
        - Information gain based on the chosen criterion.
        """
        left_idxs, right_idxs = self._split(X_column, threshold, feat_idx)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        if self.criterion == "entropy":
            return self._entropy_gain(y, left_idxs, right_idxs)
        elif self.criterion == "gini":
            return self._gini_gain(y, left_idxs, right_idxs)
        elif self.criterion == "mse_reduction":
            return self._mse_gain(y, left_idxs, right_idxs)

    def _entropy_gain(self, y, left_idxs, right_idxs):
        """
        Computes the entropy gain for a split.

        Returns:
        - Entropy gain after the split.
        """
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs) 
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) 
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy 
        return ig

    def _gini_gain(self, y, left_idxs, right_idxs):
        """
        Computes the Gini gain for a split.

        Returns:
        - Gini gain after the split.
        """
        parent_gini = self._gini(y)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return parent_gini - child_gini

    def _mse_gain(self, y, left_idxs, right_idxs):
        """
        Computes the MSE reduction for a split.

        Returns:
        - MSE reduction after the split.
        """
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        mse_parent = self._mse(y)
        mse_left = self._mse(y[left_idxs])
        mse_right = self._mse(y[right_idxs])
        weighted_mse = (n_l / n) * mse_left + (n_r / n) * mse_right
        mse_reduction = mse_parent - weighted_mse
        return mse_reduction


    def _split(self, X_column, threshold, feature_idx):
        """
        Splits data based on the threshold.

        Parameters:
        - X_column: Feature values.
        - threshold: Split threshold.
        - feature_idx: Index of the feature (to check if it's categorical).

        Returns:
        - left_idxs: Indices of the left split.
        - right_idxs: Indices of the right split.
        """
        if self.categorical_features[feature_idx]:
            left_idxs = np.argwhere(X_column == threshold).flatten()
            right_idxs = np.argwhere(X_column != threshold).flatten()
        else:
            left_idxs = np.argwhere(X_column <= threshold).flatten()
            right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy for the data
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        """Calculate the Gini index for the data"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p ** 2 for p in ps])

    def _mse(self, y):
        """
        Compute the MSE for the split using ψ(p) = sqrt(p(1 − p)).
        """
        hist = np.bincount(y, minlength=2) 
        ps = hist / len(y) 
        return np.sum([np.sqrt(p * (1 - p)) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Returns the most common label in y.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predicts class labels for the input data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Recursively traverses the tree to make a prediction.
        """
        if node.is_leaf_node():
            return node.value

        if self.categorical_features[node.feature]:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, feature_names=None, class_names=None, depth=0):
        """
        Prints the decision tree structure.

        Parameters:
        - node: Current node (starts at root if None).
        - feature_names: Optional list of feature names.
        - class_names: Optional list of class names.
        - depth: Current depth in the tree.
        """
        if node is None:
            node = self.root

        if node.is_leaf_node():
            class_label = 'p' if node.value == 1 else 'e'
            print(f"{'|   ' * depth}Leaf: Predicted class = {class_label}")
        else:
            feature_name = feature_names[node.feature] if feature_names is not None else f"Feature {node.feature}"

            if self.categorical_features[node.feature]:
                print(f"{'|   ' * depth}Node: {feature_name} == {node.threshold}")
            else:
                print(f"{'|   ' * depth}Node: {feature_name} <= {node.threshold}")
            
            if node.left is not None:
                print(f"{'|   ' * depth}---> Left:")
                self.print_tree(node.left, feature_names, class_names, depth + 1)

            if node.right is not None:
                print(f"{'|   ' * depth}---> Right:")
                self.print_tree(node.right, feature_names, class_names, depth + 1)