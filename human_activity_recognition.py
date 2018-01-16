import pandas as pd
import numpy as np
import os
import random
import argparse
from Queue import Queue
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input


class HAR:
    def __init__(self, data_folder="data", train_file="train.csv", test_file="test.csv"):
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file

        # dataframes loaded from the csv files
        self.train_data = None
        self.test_data = None
        self.n_features = None  # all columns except subject,Activity

        self.flag_encoded = False  # Set this True after encoding based on train data
        self.label_encoder = LabelEncoder()
        # ?? Should onehot dtype be changed to int?
        self.onehot_encoder = OneHotEncoder(sparse=False)

    def load_train_data(self):
        with open(os.path.join(self.data_folder, self.train_file), 'r') as fd:
            self.train_data = pd.read_csv(fd)
            self.n_features = self.train_data.shape[1] - 2

    def load_test_data(self):
        with open(os.path.join(self.data_folder, self.test_file), 'r') as fd:
            self.test_data = pd.read_csv(fd)

    def prepare_data_for_lstm(self, data, n_timestep=5):
        """Preparing data_x,data_y for input to LSTM

            Parameters
            ----------
            data : pandas dataframe (created by loading train/test data)

            Returns
            -------
            (data_x, data_y) : tuple (data point for LSTM)
        """
        n_sample = data.shape[0]
        print "sample size: {0} : feature size: {1}".format(n_sample, data.shape[1])
        data_x = []
        row_i = 0
        row_i_array = []
        # create datapoints using n_timestep
        while row_i < n_sample:
            # case #1: row_i,..,row_i+n_timestep-1 : belongs to same (subject,Activity)
            # case #2: (subject,Activity) changes before row_i+n_timestep:
            #           Check if we can create datapoint by shifting row_i back i.e. creating datapoint which overlap
            #            with previous datapoint

            # Get the labels i.e. (subject,Activity)
            cur_subject = data.ix[row_i, "subject"]
            cur_activity = data.ix[row_i, "Activity"]

            # Check for case #1
            row_j = row_i + 1
            # If case #1 satisfies then row_j - row_i = n_timestep i.e. datapoint [row_i,row_j)
            flag_sequence = True
            while (row_j < (row_i+n_timestep)) and (row_j < n_sample):
                if (data.ix[row_j, "subject"] == cur_subject) and (data.ix[row_j, "Activity"] == cur_activity):
                    row_j += 1
                else:
                    flag_sequence = False
                    break

            assert row_j-row_i <= n_timestep, "Error: while computing row_j"
            # TODO another assert to match row_j with flag_sequence

            if (row_j - row_i < n_timestep) and (row_j > row_i+1):
                # Check for case #2
                # 2nd condition i.e. row_j > row_i+1 is required to avoid getting the same datapoint as previous one
                #       This can be modified to control how much overlap we can allow
                row_k = row_i - 1

                while (row_k >= (row_j - n_timestep)) and (row_k >= 0):
                    if (data.ix[row_k, "subject"] == cur_subject) and (data.ix[row_k, "Activity"] == cur_activity):
                        row_k -= 1
                    else:
                        break

                # check if it can form datapoint
                if (row_j - row_k) == (n_timestep+1):
                    row_i = row_k + 1
                    flag_sequence = True

            if flag_sequence is True:
                # Now create the datapoint for n_timestamp: [row_i,row_j) as explained by Adam Sypniewski
                cur_datapoint = []
                for row_index in range(row_i, row_j):
                    cur_timestamp_x_data = np.array(data.ix[row_index, data.columns.difference(["subject","Activity"])])
                    cur_datapoint.append(cur_timestamp_x_data)

                data_x.append(cur_datapoint)
                row_i_array.append(row_i)

            # assigning starting row for next datapoint
            row_i = row_j

        data_x = np.array(data_x)
        # Now create the data_y
        # https://stackoverflow.com/questions/33385238/how-to-convert-pandas-single-column-data-frame-to-series-or-numpy-vector
        y_str_values = self.train_data['Activity'].values[row_i_array]
        if self.flag_encoded is False:
            y_int_encoded = self.label_encoder.fit_transform(y_str_values)
            y_int_encoded = y_int_encoded.reshape(len(y_str_values), 1)
            data_y = self.onehot_encoder.fit_transform(y_int_encoded)
            self.flag_encoded = True
        else:
            y_int_encoded = self.label_encoder.transform(y_str_values)
            y_int_encoded = y_int_encoded.reshape(len(y_str_values), 1)
            data_y = self.onehot_encoder.transform(y_int_encoded)

        return data_x, data_y

    def train_lstm_model(self, n_timestep=5, nb_epoch=5):
        """Train the LSTM model for many-to-one
        """
        # Create each datapoint with 5 timesteps (say).
        # We are doing this as each (subject,Activity) combination don't have the same number of timesteps

        # create data_x, data_y for training
        # similarly create data_x, data_y for testing

        train_data_x, train_data_y = self.prepare_data_for_lstm(self.train_data, n_timestep)
        n_class = len(train_data_y[0])
        test_data_x, test_data_y = self.prepare_data_for_lstm(self.test_data, n_timestep)

        model = Sequential()
        model.add(LSTM(output_dim=100, return_sequences=False, input_shape=(n_timestep, self.n_features)))
        model.add(Dense(output_dim=n_class, activation="sigmoid"))
        # https://keras.io/losses/
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        model.fit(x=train_data_x, y=train_data_y, nb_epoch=nb_epoch)
        scores = model.evaluate(test_data_x, test_data_y)
        print("\nAccuracy: %.2f%%" % (scores[1] * 100))

    @staticmethod
    def select_feature_subset(train_data_x, n_feature_subset, method="random"):
        """Select feature subset

            Parameters
            ----------
            n_feature_subset : py:class:`Int` : Feature subset size
            method : py:class:`String` either random or std
                    std: select features with topmost standard deviation
        """
        if method == "random":
            feature_indices = sorted(random.sample(range(train_data_x.shape[1]), n_feature_subset))
        elif method == "std":
            std_arr = np.std(a=train_data_x, axis=0)
            # https://stackoverflow.com/questions/22414152/best-way-to-initialize-and-fill-an-numpy-array
            feature_indices = np.empty(train_data_x.shape[1], dtype=int)
            j = 0
            for i, val in sorted(enumerate(std_arr), key=lambda x: x[1], reverse=True):
                feature_indices[j] = i
                j += 1
            feature_indices = feature_indices[:n_feature_subset]
        else:
            assert False, "method should be one of these: 1. random  2. std"

        return feature_indices


# TODO Currently only numeric features handled. Extend to categorical/nominal feature types.
class DecisionNode:
    def __init__(self, data_index_array, depth_level=None, parent_index=None):
        self.data_index_array = data_index_array
        self.is_terminal = None
        self.split_attribute = None
        self.split_value = None
        self.parent_index = parent_index
        self.left_child_index = None
        self.right_child_index = None
        self.depth_level = depth_level
        # predict_val: # For classification -> majority class
        #               # For regression -> average over the values in this node
        self.predict_val = None
        # class_count_dict: Only for classification
        self.class_count_dict = None


# TODO Currently only classification tree. Extend to regression.
class DecisionTree:
    def __init__(self, train_data_x, train_data_y, min_samples=5, max_depth=20, is_random_selection_predictors=False):
        """Initialize decision tree

            Parameters
            ----------
            train_data_x : numpy.ndarray of float with shape(n_samples,n_features)
            train_data_y : numpy array of string
            is_random_selection_predictors : bool
                    True for Random Forest
        """
        self.nodes = []  # Think about changes that might come due to prune
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.min_samples = min_samples  # Minimum number of samples for a node to have to be considered for split
        self.max_depth = max_depth
        self.is_random_selection_predictors = is_random_selection_predictors
        if is_random_selection_predictors:
            self.m_predictors = int(np.sqrt(train_data_x.shape[1]))
        else:
            self.m_predictors = None  # all predictors to be used for node splitting
        # TODO pass the list of feature names => probably no more required as train_data_x is build using the feature
        # names which we want

    def build_tree(self, verbose=False):
        print "Build tree started"
        root_node = DecisionNode(data_index_array=range(self.train_data_x.shape[0]), depth_level=0)
        self.nodes.append(root_node)
        self.build_sub_tree(node_index=0, verbose=verbose)
        print "Build tree completed. Number of nodes: {0}".format(len(self.nodes))

    def build_sub_tree(self, node_index, verbose=False):
        """Builds subtree recursively with node_index as root of subtree
        """
        flag_split, best_split_feature_index, best_split_feature_val, best_split_gini_index = self.split_node(node_index)
        if verbose:
            print "node_index: {0} :: flag_split: {1} : best_split_feature_index: {2} : best_split_feature_val: {3} :" \
                  " depth: {4} : gini index: {5}".format(node_index, flag_split, best_split_feature_index,
                                                         best_split_feature_val, self.nodes[node_index].depth_level,
                                                         best_split_gini_index)

        if flag_split is False:
            # assign node as node_index as terminal node
            self.nodes[node_index].is_terminal = True

            # assign the class based on majority
            class_count_dict = dict()
            for sample_i in self.nodes[node_index].data_index_array:
                cur_class = self.train_data_y[sample_i]
                if cur_class in class_count_dict:
                    class_count_dict[cur_class] += 1
                else:
                    class_count_dict[cur_class] = 1

            majority_class = None
            majority_count = 0
            for cur_class in class_count_dict:
                if class_count_dict[cur_class] > majority_count:
                    majority_class = cur_class
                    majority_count = class_count_dict[cur_class]

            assert majority_class is not None, "predicted class for the terminal node with node_index: {0} is None".format(node_index)
            self.nodes[node_index].class_count_dict = class_count_dict
            self.nodes[node_index].predict_val = majority_class
            return
        else:
            depth_level_node = self.nodes[node_index].depth_level

            # extract the indices for the children
            left_child_indices = []
            right_child_indices = []
            for sample_i in self.nodes[node_index].data_index_array:
                if self.train_data_x[sample_i, best_split_feature_index] < best_split_feature_val:
                    left_child_indices.append(sample_i)
                else:
                    right_child_indices.append(sample_i)

            if verbose:
                print "  size(left node): {0} :: size(right node): {1}".format(len(left_child_indices),
                                                                               len(right_child_indices))
            # create the nodes: left child, right child
            left_child_node = DecisionNode(data_index_array=left_child_indices, depth_level=depth_level_node+1, parent_index=node_index)
            right_child_node = DecisionNode(data_index_array=right_child_indices, depth_level=depth_level_node+1, parent_index=node_index)

            left_child_node_index = len(self.nodes)
            self.nodes.append(left_child_node)
            right_child_node_index = len(self.nodes)
            self.nodes.append(right_child_node)

            self.nodes[node_index].is_terminal = False
            self.nodes[node_index].split_attribute = best_split_feature_index
            self.nodes[node_index].split_value = best_split_feature_val
            self.nodes[node_index].left_child_index = left_child_node_index
            self.nodes[node_index].right_child_index = right_child_node_index

            # recursively call build_sub_tree with left and right children
            self.build_sub_tree(left_child_node_index)
            self.build_sub_tree(right_child_node_index)

    def split_node(self, node_index):
        """Split node into two children (if applicable). Else tag the node as terminal node.
        """
        # First check suitability of splitting the node.
        node_level = self.nodes[node_index].depth_level
        if node_level >= self.max_depth:
            self.nodes[node_index].is_terminal = True
            return False, None, None, None

        node_sample_size = len(self.nodes[node_index].data_index_array)
        if node_sample_size < self.min_samples:
            self.nodes[node_index].is_terminal = True
            return False, None, None, None

        assert node_sample_size > 0, "Empty node: node_index: {0} :: node_level: {1}".format(node_index, node_level)

        # If all the samples in the current node belongs to same class, there's no need to split the node
        # ??? Should we be strict or allow flexibitly i.e. if the dominant class is beyond a certain fraction we shouldn't split
        # But the issue could also be that we are doing a problem of unbalanced classes
        flag_mixed_class = False
        prev_sample_index = self.nodes[node_index].data_index_array[0]
        for sample_i in range(1, node_sample_size):
            cur_sample_index = self.nodes[node_index].data_index_array[sample_i]
            if self.train_data_y[prev_sample_index] != self.train_data_y[cur_sample_index]:
                flag_mixed_class = True
                break
            else:
                prev_sample_index = cur_sample_index

        if flag_mixed_class is False:
            return False, None, None, None

        # Iterate over the attributes to decide the attribute for the split
        n_features = self.train_data_x.shape[1]

        best_gini_index = np.inf
        best_split_feature_index = None
        best_split_feature_val = None

        if self.is_random_selection_predictors:
            # used in Random Forest
            features_arr = random.sample(range(n_features), self.m_predictors)
        else:
            features_arr = range(n_features)

        for feature_i in features_arr:
            cur_feature_values = self.train_data_x[self.nodes[node_index].data_index_array, feature_i]
            # ?? can we avoid sorting at each level
            # sort the feature values as we would select split points at different percentile
            cur_feature_values = sorted(cur_feature_values)
            # approach #1: select split points at equal interval of index
            percentile_interval = 10
            index_interval = max(1, int(np.floor(percentile_interval*len(cur_feature_values)/100)))

            for perc_i in range(1, min(int(np.floor(100/percentile_interval)), node_sample_size)):
                split_index = perc_i*index_interval - 1
                split_value = cur_feature_values[split_index]
                # split the samples into two groups based on the values of feature_i
                left_group = []
                right_group = []
                for sample_i in self.nodes[node_index].data_index_array:
                    cur_feature_value = self.train_data_x[sample_i, feature_i]
                    if cur_feature_value < split_value:
                        left_group.append(sample_i)
                    else:
                        right_group.append(sample_i)

                # compute the gini index for each of the groups
                gini_index_left_child = self.compute_gini_index(left_group)
                gini_index_right_child = self.compute_gini_index(right_group)
                gini_index_node = gini_index_left_child*len(left_group)*1.0/node_sample_size + \
                                  gini_index_right_child*len(right_group)*1.0/node_sample_size

                if gini_index_node < best_gini_index:
                    best_gini_index = gini_index_node
                    best_split_feature_index = feature_i
                    best_split_feature_val = split_value

        return True, best_split_feature_index, best_split_feature_val, best_gini_index

    def prune_tree(self):
        """Prune tree
        """
        print "tree pruning started"
        # Collect the internal nodes whose both children are leaf nodes
        candidate_node_indices = Queue()
        n_leaf_nodes = 0
        for node_index in range(len(self.nodes)):
            if self.nodes[node_index].is_terminal is True:
                # current node is itself a leaf node
                n_leaf_nodes += 1
                continue

            left_child_node_index = self.nodes[node_index].left_child_index
            right_child_node_index = self.nodes[node_index].right_child_index

            if self.nodes[left_child_node_index].is_terminal and self.nodes[right_child_node_index].is_terminal:
                candidate_node_indices.put(node_index)

        while candidate_node_indices.empty() is False:
            node_index = candidate_node_indices.get()
            assert self.nodes[node_index].is_terminal is False, "Only internal nodes should come"

            # check if converting sub-tree with node_index as root would have less total cost than the current total cost
            left_child_node_index = self.nodes[node_index].left_child_index
            right_child_node_index = self.nodes[node_index].right_child_index

            left_child_class_count_dict = self.nodes[left_child_node_index].class_count_dict
            right_child_class_count_dict = self.nodes[right_child_node_index].class_count_dict

            # combined class count dict
            combined_class_count_dict = left_child_class_count_dict
            for cur_class in right_child_class_count_dict:
                if cur_class in combined_class_count_dict:
                    combined_class_count_dict[cur_class] += right_child_class_count_dict[cur_class]
                else:
                    combined_class_count_dict[cur_class] = right_child_class_count_dict[cur_class]

            combined_majority_class = None
            combined_majority_count = 0
            for cur_class in combined_class_count_dict:
                if combined_class_count_dict[cur_class] > combined_majority_count:
                    combined_majority_class = cur_class
                    combined_majority_count = combined_class_count_dict[cur_class]

            left_child_majority_class = self.nodes[left_child_node_index].predict_val
            right_child_majority_class = self.nodes[right_child_node_index].predict_val

            combined_node_classification_error = len(self.nodes[left_child_node_index].data_index_array) + \
                                                 len(self.nodes[right_child_node_index].data_index_array) - \
                                                 combined_majority_count
            subtree_classification_error = len(self.nodes[left_child_node_index].data_index_array) - \
                                           left_child_class_count_dict[self.nodes[left_child_node_index].predict_val] +\
                                           len(self.nodes[right_child_node_index].data_index_array) - \
                                           right_child_class_count_dict[self.nodes[right_child_node_index].predict_val]

            # As both the left and right nodes are leaf nodes, pruning them would lead to one decrease in number of leaf nodes
            n_samples = len(self.nodes[0].data_index_array)
            flag_prune = False
            # TODO threshold for classification error increase should be passed as input parameter \
            # This should be treated as one of the hyper-parameters
            if (combined_node_classification_error - subtree_classification_error)*1.0/n_samples < 0.01:
                flag_prune = True

            if flag_prune is False:
                continue

            # if yes, then convert this sub-tree into a leaf node
            self.nodes[left_child_node_index] = None
            self.nodes[right_child_node_index] = None

            self.nodes[node_index].is_terminal = True
            self.nodes[node_index].predict_val = combined_majority_class
            self.nodes[node_index].class_count_dict = combined_class_count_dict

            # Also if other child of the parent of node_index is leaf node then push the parent into candidate_node_indices
            parent_node_index = self.nodes[node_index].parent_index
            if parent_node_index == 0:
                # root node can't be pruned
                continue
            if self.nodes[parent_node_index].left_child_index == node_index:
                other_child_node_index = self.nodes[parent_node_index].right_child_index
            else:
                other_child_node_index = self.nodes[parent_node_index].left_child_index

            if self.nodes[other_child_node_index].is_terminal is True:
                candidate_node_indices.put(parent_node_index)

        n_pruned_nodes = 0
        for node_index in range(len(self.nodes)):
            if self.nodes[node_index] is None:
                n_pruned_nodes += 1
        print "{0} nodes pruned out of {1}".format(n_pruned_nodes, len(self.nodes))

    def compute_gini_index(self, group_samples):
        """Compute gini index for the group which we are considering for child node

            Parameters
            ----------
            group_samples : numpy array of indices of the train data
        """
        class_count_dict = dict()
        # populate count of samples in the group belonging to its corresponding class
        for sample_i in group_samples:
            cur_class = self.train_data_y[sample_i]
            if cur_class in class_count_dict:
                class_count_dict[cur_class] += 1
            else:
                class_count_dict[cur_class] = 1

        # Now compute the gini index
        n_group_samples = len(group_samples)
        gini_index = 0.0
        for class_y in class_count_dict:
            cur_class_prob = class_count_dict[class_y]*1.0/n_group_samples
            gini_index += cur_class_prob*(1.0 - cur_class_prob)

        return gini_index

    def evaluate(self, test_data_x, test_data_y):
        accuracy = 0.0
        names_class = np.unique(self.train_data_y)
        # https://stackoverflow.com/questions/11106536/adding-row-column-headers-to-numpy-matrices (bmu's answer)
        confusion_matrix = pd.DataFrame(np.zeros((len(names_class), len(names_class))), index=names_class, columns=names_class)
        for sample_i in range(test_data_x.shape[0]):
            test_sample_x = test_data_x[sample_i, :]
            true_class = test_data_y[sample_i]
            predicted_class = self.predict(test_sample_x)

            if predicted_class == true_class:
                accuracy += 1

            confusion_matrix[true_class][predicted_class] += 1

        accuracy /= test_data_x.shape[0]
        accuracy *= 100
        print "accuracy: ", accuracy
        print "confusion matrix: (row=>true class; col=>predicted class)"
        print confusion_matrix

    def predict(self, test_sample_x):
        # parse the tree till we reach the terminal node
        node_index = 0  # root node
        while not self.nodes[node_index].is_terminal:
            split_attribute = self.nodes[node_index].split_attribute
            split_value = self.nodes[node_index].split_value
            if test_sample_x[split_attribute] < split_value:
                node_index = self.nodes[node_index].left_child_index
            else:
                node_index = self.nodes[node_index].right_child_index

        assert self.nodes[node_index].is_terminal is True, "reached a non-terminal node: node_index: {0}".format(node_index)
        return self.nodes[node_index].predict_val


# TODO currently only works for classification
class Bagging:
    """Bagging using boot strapped samples
    """
    def __init__(self, train_data_x, train_data_y, n_decision_trees, min_samples=5, max_depth=20,
                 is_random_selection_predictors=False):
        self.decision_trees = []
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.n_decision_trees = n_decision_trees
        self.min_samples = min_samples  # Minimum number of samples for a node to have to be considered for split
        self.max_depth = max_depth
        # False: Bagging
        # True: Random Forest
        self.is_random_selection_predictors = is_random_selection_predictors

    def build_trees(self):
        n_sample = self.train_data_x.shape[0]
        for tree_i in range(self.n_decision_trees):
            print "tree #{0}".format(tree_i)
            # bootstrap samples with replacement
            sample_indices = np.random.choice(a=range(n_sample), size=n_sample, replace=True)
            cur_train_data_x = self.train_data_x[sample_indices, :]
            cur_train_data_y = self.train_data_y[sample_indices]
            # create a decision tree using the current bootstrapped samples
            decision_tree_obj = DecisionTree(train_data_x=cur_train_data_x, train_data_y=cur_train_data_y,
                                             min_samples=self.min_samples, max_depth=self.max_depth,
                                             is_random_selection_predictors=self.is_random_selection_predictors)
            decision_tree_obj.build_tree()
            self.decision_trees.append(decision_tree_obj)

    def evaluate(self, test_data_x, test_data_y):
        accuracy = 0.0
        names_class = np.unique(self.train_data_y)
        # https://stackoverflow.com/questions/11106536/adding-row-column-headers-to-numpy-matrices (bmu's answer)
        confusion_matrix = pd.DataFrame(np.zeros((len(names_class), len(names_class))), index=names_class, columns=names_class)
        for sample_i in range(test_data_x.shape[0]):
            test_sample_x = test_data_x[sample_i, :]
            true_class = test_data_y[sample_i]
            predicted_class = self.predict(test_sample_x)

            if predicted_class == true_class:
                accuracy += 1

            confusion_matrix[true_class][predicted_class] += 1

        accuracy /= test_data_x.shape[0]
        accuracy *= 100
        print "accuracy: ", accuracy
        print "confusion matrix: (row=>true class; col=>predicted class)"
        print confusion_matrix

    def predict(self, test_sample_x):
        """Predict based on majority vote(for classification)
        """
        predicted_class_dict = dict()
        for tree_i in range(self.n_decision_trees):
            predicted_class = self.decision_trees[tree_i].predict(test_sample_x)
            if predicted_class in predicted_class_dict:
                predicted_class_dict[predicted_class] += 1
            else:
                predicted_class_dict[predicted_class] = 1

        count_majority_class = 0
        majority_class = None

        for predicted_class in predicted_class_dict:
            if predicted_class_dict[predicted_class] > count_majority_class:
                majority_class = predicted_class
                count_majority_class = predicted_class_dict[predicted_class]

        return majority_class


def process(method):
    har_obj = HAR()
    har_obj.load_train_data()
    har_obj.load_test_data()

    if method == "lstm":
        har_obj.train_lstm_model(n_timestep=10, nb_epoch=5)
    elif method == "decision tree":
        # Decision Tree
        train_data_x = np.array(har_obj.train_data.ix[:, har_obj.train_data.columns.difference(["subject", "Activity"])])
        train_data_y = np.array(har_obj.train_data.ix[:, "Activity"])
        # Pick either a) random subset of the features  b) features with most variance
        feature_index_arr = har_obj.select_feature_subset(train_data_x=train_data_x, n_feature_subset=100, method="std")
        train_data_x = train_data_x[:, feature_index_arr]
        decision_tree_obj = DecisionTree(train_data_x, train_data_y, max_depth=20)
        decision_tree_obj.build_tree(verbose=True)
        decision_tree_obj.prune_tree()
        test_data_x = np.array(har_obj.test_data.ix[:, har_obj.test_data.columns.difference(["subject", "Activity"])])
        test_data_y = np.array(har_obj.test_data.ix[:, "Activity"])
        test_data_x = test_data_x[:, feature_index_arr]
        decision_tree_obj.evaluate(test_data_x, test_data_y)
    elif method == "bagging" or method == "random forest":
        # Bagging or Bootstrap aggregation, Random Forest
        train_data_x = np.array(
            har_obj.train_data.ix[:, har_obj.train_data.columns.difference(["subject", "Activity"])])
        train_data_y = np.array(har_obj.train_data.ix[:, "Activity"])
        # Pick either a) random subset of the features  b) features with most variance
        feature_index_arr = har_obj.select_feature_subset(train_data_x=train_data_x, n_feature_subset=200, method="std")
        train_data_x = train_data_x[:, feature_index_arr]
        if method == "bagging":
            flag_random_selection_predictors = False
        else:
            flag_random_selection_predictors = True
        bagging_obj = Bagging(train_data_x, train_data_y, n_decision_trees=11,
                              is_random_selection_predictors=flag_random_selection_predictors)
        bagging_obj.build_trees()
        test_data_x = np.array(har_obj.test_data.ix[:, har_obj.test_data.columns.difference(["subject", "Activity"])])
        test_data_y = np.array(har_obj.test_data.ix[:, "Activity"])
        test_data_x = test_data_x[:, feature_index_arr]
        bagging_obj.evaluate(test_data_x, test_data_y)
    else:
        assert False, "undefined method: {0}".format(method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-m", "--method", type=str)
    args = parser.parse_args()
    process(args.method)

"""
How to run:
    Example:
    python human_activity_recognition.py --method "decision tree"

References:
    Followed the LSTM explanation in these forums:
    https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features
    https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

    One-hot encoding:
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    step #1: encode string to integer
    step #2: one-hot encode integer to one-hot
    Mentions that sparse encoding might not be suitable for Keras

    https://stackoverflow.com/questions/33385238/how-to-convert-pandas-single-column-data-frame-to-series-or-numpy-vector

    Decision Tree:
        https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""