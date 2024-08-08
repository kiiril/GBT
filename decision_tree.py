import numpy as np

# ID3
class ID3():
    def __init__(self, mode="cls", max_depth=None):
        self.root = None
        self.mode = mode
        self.max_depth = max_depth
    
    def entropy(self, feature: np.array) -> float:
        entropy = 0
        values, counts = np.unique(feature, return_counts=True)
        total_count = np.sum(counts)
        for i in range(len(values)):
            p = counts[i] / total_count
            entropy -= p * np.log2(p)
        return entropy
    
    def variance(self, y: np.array) -> float:
        return np.var(y)

    
    def info_gain(self, X: np.array, feature_index: int, y: np.array) -> float:

        if self.mode == "cls":
            info_before_split = self.entropy(y)
        else:
            info_before_split = self.variance(y)

        values, counts = np.unique(X[:, feature_index], return_counts=True)
        total_count = np.sum(counts)
        
        info = 0
        for i in range(len(values)):
            subset = y[X[:, feature_index] == values[i]]

            if self.mode == "cls":
                info += (counts[i] / total_count) * self.entropy(subset)
            else:
                info += (counts[i] / total_count) * self.variance(subset)
            
        return info_before_split - info

    
    def split(self, X, feature_index, param, y):
        cond = X[:, feature_index] == param
        return X[cond], y[cond]
    

    def fit(self, X: np.array, y: np.array):
        self.root = self._fit(X, y)
        return self

    def _fit(self, X: np.array, y: np.array, depth=0):
        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth):
            
            if self.mode == "cls":
                return Node(value=np.unique(y)[0]) # Return the majority class
            else:
                return Node(value=np.mean(y))

        features_info_gain = [self.info_gain(X, feature_index, y) for feature_index in range(np.shape(X)[1])]
        
        best_feature_index = np.argmax(features_info_gain)
        
        root = Node(feature_index=best_feature_index)
        
        for param in np.unique(X[:, best_feature_index]):
            # data after the split
            sub_X, sub_y = self.split(X, best_feature_index, param, y)
            # recursive call
            branch = self._fit(sub_X, sub_y, depth + 1)
            # append branch to the tree
            root.add_child((param, branch))
            
        return root

    def predict_one(self, node, instance):
        if node.value is not None:
            return node.value
        
        feature_value = instance[node.feature_index]

        if self.mode == 'reg':
        #     # Regression prediction
        #     child_values = np.array([child.value for param, child in node.children if param == feature_value])
        #     if child_values.size > 0:
        #         return child_values.mean()
        #     else:
        #         # If no matching child node, return the average of all child node values
        #         all_child_values = np.array([child.value for _, child in node.children])
        #         return all_child_values.mean()
            # Regression prediction
            child_values = [child.value for param, child in node.children if param == feature_value and child.value is not None]
            if child_values:
                return sum(child_values) / len(child_values)
            else:
                # If no matching child node, return the average of all non-None child node values
                all_child_values = [child.value for _, child in node.children if child.value is not None]
                if all_child_values:
                    return sum(all_child_values) / len(all_child_values)
                else:
                    # If all child node values are None, return None
                    return None
        else:
            # Classification prediction
            for param, child in node.children:
                if feature_value == param:
                    return self.predict_one(child, instance)

            child_values = [child.value for param, child in node.children]
            return max(set(child_values), key=child_values.count)

    def predict(self, X):
        return np.array([self.predict_one(self.root, instance) for instance in X])
        

class Node():
    def __init__(self, feature_index=None, value=None):
        self.feature_index = feature_index
        self.value = value # For classification, this is the class; for regression, this is a continuous value
        self.children = []

    def add_child(self, node) -> None:
        self.children.append(node)