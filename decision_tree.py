import numpy as np

# ID3
class ID3():
    def __init__(self, max_depth=None, min_samples_split=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    

    def entropy(self, feature: np.array) -> float:
        entropy = 0
        values, counts = np.unique(feature, return_counts=True)
        total_count = np.sum(counts)
        for i in range(len(values)):
            p = counts[i] / total_count
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy


    def info_gain(self, X: np.array, feature_index: int, y: np.array) -> tuple:
        if np.issubdtype(X[:, feature_index].dtype, np.number):
            return self.info_gain_continuous(X, feature_index, y)
        else:
            return self.info_gain_discrete(X, feature_index, y), None


    def info_gain_continuous(self, X: np.array, feature_index: int, y: np.array) -> float:
        info_before_split = self.entropy(y)
        sorted_indices = np.argsort(X[:, feature_index])
        sorted_x = X[sorted_indices, feature_index]
        sorted_y = y[sorted_indices]
        
        best_info_gain = 0
        best_threshold = None
        
        for i in range(1, len(sorted_x)):
            if sorted_x[i] != sorted_x[i-1]:
                threshold = (sorted_x[i] + sorted_x[i-1]) / 2
                left_y = sorted_y[:i]
                right_y = sorted_y[i:]
                
                info_left = self.entropy(left_y)
                info_right = self.entropy(right_y)
                
                info = (len(left_y) / len(y)) * info_left + (len(right_y) / len(y)) * info_right
                info_gain = info_before_split - info
                
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_threshold = threshold
        
        return best_info_gain, best_threshold
    
    
    def info_gain_discrete(self, X: np.array, feature_index: int, y: np.array) -> float:
        info_before_split = self.entropy(y)

        values, counts = np.unique(X[:, feature_index], return_counts=True)
        total_count = np.sum(counts)
        
        info = 0
        for i in range(len(values)):
            subset = y[X[:, feature_index] == values[i]]
            info += (counts[i] / total_count) * self.entropy(subset)

        return info_before_split - info

    
    def split(self, X: np.array, feature_index: int, threshold: str, y: np.array):
        if np.issubdtype(X[:, feature_index].dtype, np.number):
            cond = X[:, feature_index] <= threshold
        else:
            cond = X[:, feature_index] == threshold
        return X[cond], y[cond], X[~cond], y[~cond]
    

    def fit(self, X: np.array, y: np.array):
        self.root = self._fit(X, y)
        return self


    def _fit(self, X: np.array, y: np.array, depth=0):
        n_samples, n_features = X.shape
        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth) or (n_samples < self.min_samples_split):
            unique_y, counts = np.unique(y, return_counts=True)
            return Node(value=unique_y[np.argmax(counts)]) # return the majority class or remaining class
        
        features_info_gain = [self.info_gain(X, feature_index, y) for feature_index in range(n_)]
        
        best_feature_index = np.argmax([gain for gain, _ in features_info_gain])

        best_info_gain, best_threshold = features_info_gain[best_feature_index]
        
        root = Node(feature_index=best_feature_index, threshold=best_threshold)

        if np.issubdtype(X[:, best_feature_index].dtype, np.number):
            left_X, left_y, right_X, right_y = self.split(X, best_feature_index, best_threshold, y)
            root.add_child(('<=', self._fit(left_X, left_y, depth + 1)))
            root.add_child(('>', self._fit(right_X, right_y, depth + 1)))
        else:
            for param in np.unique(X[:, best_feature_index]):
                sub_X, sub_y, _, _ = self.split(X, best_feature_index, param, y)
                branch = self._fit(sub_X, sub_y, depth + 1)
                root.add_child((param, branch))
        return root


    def predict_one(self, node, sample):
        if node.value is not None:
            return node.value
        
        feature_value = sample[node.feature_index]

        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self.predict_one(node.children[0][1], sample)
            else:
                return self.predict_one(node.children[1][1], sample)
        else:
            for param, child in node.children:
                if feature_value == param:
                    return self.predict_one(child, sample)

        # If no matching branch is found, return the majority class at the current node
        unique_y, counts = np.unique([child.value for param, child in node.children if child.value is not None], return_counts=True)
        if len(unique_y) > 0:
            return unique_y[np.argmax(counts)]
        else:
            return None  # if no valid class exists, this should theoretically never happen if the tree is well-formed

    def predict(self, X: np.array) -> np.array:
        return [self.predict_one(self.root, sample) for sample in X]
        

class Node():
    def __init__(self, feature_index=None, value=None, threshold=None):
        self.feature_index = feature_index
        self.value = value
        self.threshold = threshold
        self.children = []

    def add_child(self, node) -> None:
        self.children.append(node)