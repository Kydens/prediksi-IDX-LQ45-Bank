import pandas as pd
import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class Node:
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None, left: Optional['Node'] = None, right: Optional['Node'] = None, value: Optional[float] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth: int = None, max_features: float = None, min_samples_leaf: float = 1, min_samples_split: float = 2):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None
        
    def _calculate_SSR(self, y: np.ndarray)->float:
        if len(y) == 0:
            return 0
        
        mean = np.mean(y)
        return np.sum((y - mean)**2)
    
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray)->Tuple[float, int, float]:
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features_to_consider = self.max_features or X.shape[1]
        feature_indices = np.random.choice(X.shape[1], n_features_to_consider, replace=False)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if(np.sum(left_mask) < self.min_samples_leaf
                   or np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                left_ssr = self._calculate_SSR(y[left_mask])
                right_ssr = self._calculate_SSR(y[right_mask])
                total_ssr = left_ssr + right_ssr
                
                if total_ssr < best_score:
                    best_score = total_ssr
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_score, best_feature, best_threshold
    
    
    def _build_tree_rf(self, X: np.ndarray, y: np.ndarray, depth: int = 0)->Node:
        n_samples = X.shape[0]
        
        if(self.max_depth is not None and depth >= self.max_depth
           or n_samples < self.min_samples_split 
           or n_samples < 2 * self.min_samples_leaf
           or len(np.unique(y)) == 1):
            return Node(value=np.mean(y))
        
        _, best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=np.mean(y))
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree_rf(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree_rf(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = int(X.shape[1])
        if self.max_features > self.n_features:
            self.max_features = self.n_features
        
        self.root = self._build_tree_rf(X, y)
        return self
    
    
    def _predict_single(self, X: np.ndarray, node: Node)->float:
        if node.value is not None:
            return node.value
        if X[node.feature_idx] <= node.threshold:
            return self._predict_single(X, node.left)

        return self._predict_single(X, node.right)
    
    def predict(self, X: np.ndarray)->np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])
    
    
class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, max_features: float = 1, min_samples_leaf: float = 1, min_samples_split: float = 2, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.trees = []
        
    
    def _bootstrap_sample(self, X:np.ndarray, y:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    
    def _fit_tree(self, X: np.ndarray, y: np.ndarray)->DecisionTree:
        X_boot, y_boot = self._bootstrap_sample(X, y)
        
        tree = DecisionTree(self.max_depth, self.max_features, self.min_samples_leaf, self.min_samples_split)
        
        return tree.fit(X_boot, y_boot)
    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_jobs = self.n_jobs if self.n_jobs > 0 else None
        
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            self.trees = list(exe.map(
                lambda _: self._fit_tree(X, y),
                range(self.n_estimators)
            ))
            
        return self
    
    
    def predict(self, X: np.ndarray)->np.ndarray:
        pred = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(pred, axis=0)


class XGBoostTree:
    def __init__(self, n_estimators: int = 100, eta: float = 0.3, max_depth: int = 6, subsample: float = 1, min_child_weight: float = 1, lambda_: float = 1):
        self.n_estimators = n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.root = None
        
        
    def _calc_gain(self, gradient: np.ndarray, hessian: np.ndarray)->float:
        grad = np.sum(gradient)
        hess = np.sum(hessian)
        
        return ((grad * grad) / (hess + self.lambda_))


    def _find_best_splits(self, X: np.ndarray, gradient: np.ndarray, hessian: np.ndarray)->Tuple[float, int, float]:
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_gain = self._calc_gain(gradient[left_mask], hessian[left_mask])
                right_gain = self._calc_gain(gradient[right_mask], hessian[right_mask])
                total_gain = left_gain + right_gain
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_gain, best_feature, best_threshold
    
    
    def _calc_weight(self, gradient: np.ndarray, hessian: np.ndarray)->float:
        return ((-np.sum(gradient)) / (np.sum(hessian) + self.lambda_))
    
    
    def _build_tree_xgb(self, X: np.ndarray, gradient: np.ndarray, hessian: np.ndarray, depth: int = 0)->Node:
        n_samples = X.shape[0]
        weight = self._calc_weight(gradient, hessian)
        
        if(depth >= self.max_depth or n_samples < self.min_child_weight
           or np.sum(hessian) < self.min_child_weight):
            return Node(value=weight * self.eta)
        
        gain, feature_idx, threshold = self._find_best_splits(X, gradient, hessian)
        
        if feature_idx is None:
            return Node(value=weight * self.eta)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree_xgb(X[left_mask], gradient[left_mask], hessian[left_mask], depth + 1)
        right_child = self._build_tree_xgb(X[right_mask], gradient
        [right_mask], hessian[right_mask], depth + 1)

        return Node(feature_idx, threshold, left_child, right_child)
    
    
    def fit(self, X: np.ndarray, gradient: np.ndarray, hessian: np.ndarray):
        grad = gradient
        hess = hessian
        
        if self.subsample < 1:
            n_samples = int(X.shape[0] * self.subsample)
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X = X[indices]
            grad = gradient[indices]
            hess = hessian[indices]
        
        self.root = self._build_tree_xgb(X, grad, hess)
        return self
    
    
    def _predict_single(self, X: np.ndarray, node: Node)->float:
        if node.value is not None:
            return node.value
        
        if X[node.feature_idx] <= node.threshold:
            return self._predict_single(X, node.left)
        
        return self._predict_single(X, node.right)
        
    
    def predict(self, X: np.ndarray)->np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])
        

class XGBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 100, eta: float = 0.3, max_depth: int = 6, subsample: float = 1, min_child_weight: float = 1, lambda_: float = 1):
        self.n_estimators = n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.trees = []
        self.base_score = np.ndarray = np.array([])
        
    
    def _compute_gradient_hessian(self, y_true: np.ndarray, y_pred: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        grad = 2 * (y_pred - y_true)
        hess = np.ones_like(y_true) * 2
                
        return grad, hess
                
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_score = np.mean(y)
        y_pred = np.full_like(y, self.base_score)
        
        for _ in range(self.n_estimators):
            grad, hess = self._compute_gradient_hessian(y, y_pred)
            
            tree = XGBoostTree(
                max_depth = self.max_depth,
                min_child_weight = self.min_child_weight,
                eta = self.eta,
                subsample = self.subsample,
                lambda_ = self.lambda_
            )
            
            tree.fit(X, grad, hess)
            self.trees.append(tree)
            
            y_pred += tree.predict(X)
            
        return self
        
    
    def predict(self, X: np.ndarray)->np.ndarray:
        y_pred = np.full((X.shape[0],), self.base_score)
        
        for tree in self.trees:
            y_pred += tree.predict(X)
        
        return y_pred
    
    
if __name__ == "__main__":
    # Generate a dataset with pandas
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        "feature_1": np.random.randn(data_size),
        "feature_2": np.random.randn(data_size),
        "feature_3": np.random.randn(data_size),
        "feature_4": np.random.randn(data_size),
        "feature_5": np.random.randn(data_size),
    })
    
    df["target"] = (df["feature_1"] * 2 + df["feature_2"] - df["feature_3"] * 0.5 
                    + np.sin(df["feature_4"]) + np.random.randn(data_size) * 0.1)
    
    # Convert DataFrame to numpy arrays for compatibility
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # Split the data
    train_idx = np.random.choice([True, False], size=len(X), p=[0.8, 0.2])
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    
    # Initialize RandomForestRegressor and fit it
    rf = RandomForestRegressor(
        n_estimators=10,  # Reduced for speed in large tests
        max_depth=5,
        max_features=3,
        min_samples_leaf=2,
        min_samples_split=5
    )
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    
    # Output sample predictions for verification
    print("Random Forest Predictions (Sample):", rf_predictions[:10])

    # Initialize XGBoostRegressor and fit it
    xgb = XGBoostRegressor(
        n_estimators=10,  # Reduced for speed in large tests
        eta=0.1,
        max_depth=3,
        subsample=0.8,
        min_child_weight=1,
        lambda_=1
    )
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)
    
    # Output sample predictions for verification
    print("XGBoost Predictions (Sample):", xgb_predictions[:10])