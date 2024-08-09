import numpy as np

# ID3
class ID3():
    def __init__(self, max_depth: int):
        self.root = None
        self.max_depth = max_depth
    

    def entropy(self, feature: np.array) -> float:
        entropy = 0
        values, counts = np.unique(feature, return_counts=True)
        total_count = np.sum(counts)
        for i in range(len(values)):
            p = counts[i] / total_count
            entropy -= p * np.log2(p)
        return entropy
    
    
    def info_gain(self, X: np.array, feature_index: int, y: np.array) -> float:
        info_before_split = self.entropy(y)

        values, counts = np.unique(X[:, feature_index], return_counts=True)
        total_count = np.sum(counts)
        
        info = 0
        for i in range(len(values)):
            subset = y[X[:, feature_index] == values[i]]
            info += (counts[i] / total_count) * self.entropy(subset)

        return info_before_split - info

    
    def split(self, X: np.array, feature_index: int, param: str, y: np.array):
        cond = X[:, feature_index] == param
        return X[cond], y[cond]
    

    def fit(self, X: np.array, y: np.array):
        self.root = self._fit(X, y)
        return self


    def _fit(self, X: np.array, y: np.array, depth=0):
        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth):
            unique_y, counts = np.unique(y, return_counts=True)
            return Node(value=unique_y[np.argmax(counts)]) # return the majority class or remaining class
        
        features_info_gain = [self.info_gain(X, feature_index, y) for feature_index in range(np.shape(X)[1])]
        
        best_feature_index = np.argmax(features_info_gain)
        
        root = Node(feature_index=best_feature_index)
        
        for param in np.unique(X[:, best_feature_index]):
            sub_X, sub_y = self.split(X, best_feature_index, param, y)
            branch = self._fit(sub_X, sub_y, depth + 1)
            root.add_child((param, branch))
            
        return root


    def predict_one(self, node, instance):
        if node.value is not None:
            return node.value
        
        feature_value = instance[node.feature_index]
        
        for param, child in node.children:
            if feature_value == param:
                return self.predict_one(child, instance)

        # If no matching branch is found, return the majority class at the current node
        unique_y, counts = np.unique([child.value for param, child in node.children if child.value is not None], return_counts=True)
        if len(unique_y) > 0:
            return unique_y[np.argmax(counts)]
        else:
            return None  # If no valid class exists, this should theoretically never happen if the tree is well-formed.

    def predict(self, X: np.array) -> np.array:
        return [self.predict_one(self.root, instance) for instance in X]
        

class Node():
    def __init__(self, feature_index=None, value=None):
        self.feature_index = feature_index
        self.value = value
        self.children = []

    def add_child(self, node) -> None:
        self.children.append(node)