import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, column=None, value=None, true_branch=None, false_branch=None, current_results=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.current_results = current_results

def divideset(rows, column, value):
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

def entropy(rows):
    from math import log2
    results = Counter([row[-1] for row in rows])
    ent = 0.0
    for r in results.values():
        p = float(r) / len(rows)
        ent = ent - p * log2(p)
    return ent

def build_tree(rows):
    if len(rows) == 0: return DecisionNode()
    current_score = entropy(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1  # Last column is the target value
    for col in range(0, column_count):
        column_values = {row[col] for row in rows}
        for value in column_values:
            set1, set2 = divideset(rows, col, value)
            p = float(len(set1)) / len(rows)
            gain = current_score - p * entropy(set1) - (1 - p) * entropy(set2)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return DecisionNode(column=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)
    else:
        return DecisionNode(current_results=Counter([row[-1] for row in rows]))

def classify(observation, tree):
    if tree.current_results is not None:
        return tree.current_results
    else:
        v = observation[tree.column]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        else:
            if v == tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        return classify(observation, branch)

def print_tree(node, spacing=""):
    """Recursively print the decision tree."""

    # Base case: we've reached a leaf
    if node.current_results is not None:
        print(spacing + "Predict", node.current_results)
        return

    # Print the question at this node
    print(spacing + f"Is {node.column} == {node.value}?")

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")



# # Sample1 usage:
# 
# # Data: [height, hair_length, voice_pitch, gender]
# # Note: This is a toy dataset for illustration purposes
# data = [
#     [6, 5, 7, 'male'],
#     [5.5, 4, 6, 'male'],
#     [5.7, 4.5, 5, 'male'],
#     [5.2, 3.5, 4, 'female'],
#     [5, 3, 3, 'female'],
#     [5.5, 3.7, 3.5, 'female']
# ]
# 
# tree = build_tree(data)
# print(classify([6, 5, 7], tree))
# print(classify([5, 3, 3], tree))

class J48DecisionTree:
    def __init__(self):
        self.tree = None
        
    def fit(self, X, y):
        # Combine X and y for the format expected by build_tree function
        data = [list(X[i]) + [y[i]] for i in range(len(X))]
        self.tree = build_tree(data)

    def predict(self, X):
        # Predict each instance
        predictions = [classify(instance, self.tree) for instance in X]
        return predictions



if __name__ == "__main__":
    data = [
        ['sunny', 'hot', 'high', 'FALSE', 'no'],
        ['sunny', 'hot', 'high', 'TRUE', 'no'],
        ['overcast', 'hot', 'high', 'FALSE', 'yes'],
        ['rainy', 'mild', 'high', 'FALSE', 'yes'],
        ['rainy', 'cool', 'normal', 'FALSE', 'yes'],
        ['rainy', 'cool', 'normal', 'TRUE', 'no'],
        ['overcast', 'cool', 'normal', 'TRUE', 'yes'],
        ['sunny', 'mild', 'high', 'FALSE', 'no'],
        ['sunny', 'cool', 'normal', 'FALSE', 'yes'],
        ['rainy', 'mild', 'normal', 'FALSE', 'yes'],
        ['sunny', 'mild', 'normal', 'TRUE', 'yes'],
        ['overcast', 'mild', 'high', 'TRUE', 'yes'],
        ['overcast', 'hot', 'normal', 'FALSE', 'yes'],
        ['rainy', 'mild', 'high', 'TRUE', 'no']
    ]

   


    # Build the decision tree using the weather dataset
    tree = build_tree(data)
    print_tree(tree)
    # Classify a couple of new data instances to see the results
    result1 = classify(['sunny', 'mild', 'high', 'TRUE'], tree)
    print(f"Classification result for ['sunny', 'mild', 'high', 'TRUE']: {result1}")

    result2 = classify(['rainy', 'cool', 'normal', 'FALSE'], tree)
    print(f"Classification result for ['rainy', 'cool', 'normal', 'FALSE']: {result2}")


