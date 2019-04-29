from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.metrics import multiclass_auc
from tensorflow_addons.utils import test_utils


# =========================================
# Numpy solution
# =========================================
def separability(labels, probabilities, c1=0, c2=1):
    """Compute separability between classes labeled as `c1` and `c2`"""
    # Get a list of indices for class1 and class2
    indices_selected_classes = np.where(
        np.logical_or(labels == float(c1), labels == float(c2)))
    # Use above indices to select corresponding labels and probabilities
    labels_selected_classes = np.take(labels, indices_selected_classes)[0]
    probs_selected_classes = np.take(
        probabilities, indices_selected_classes, axis=0)[0]
    # Concatenate labels and probabilities
    points_selected_classes = np.concatenate(
        (probs_selected_classes, np.array([labels_selected_classes]).T),
        axis=1)
    # Compute rank for each data point
    all_rank_indices = np.arange(1, np.shape(labels_selected_classes)[0] + 1)
    # Compute total ranks for class1 and class2
    ranks_class1 = ranks_for_class_np(points_selected_classes, c1,
                                      all_rank_indices)
    ranks_class2 = ranks_for_class_np(points_selected_classes, c2,
                                      all_rank_indices)
    # Number of `c1` and `c2` instances
    n1, n2 = np.shape(ranks_class1)[0], np.shape(ranks_class2)[0]
    # Eqn 3 with 1->2 and 0->1 and averaged over swapped classes
    return (sum(ranks_class1) + sum(ranks_class2) - (n1 * (n1 + 1) / 2.0) -
            (n2 * (n2 + 1) / 2.0)) / (2 * float(n1 * n2))


def ranks_for_class_np(points, c, all_rank_indices):
    """Compute ranks for class `c`"""
    # Sort by class `c` probability
    points_selected_sorted_by_prob_label = points[points[:, c].argsort()]
    # Now we need only labels - drop probabilities
    labels_sorted_by_prob_label = points_selected_sorted_by_prob_label[:, -1]
    # Labels of class `c`
    rank_indices_label = np.where(labels_sorted_by_prob_label == c)
    # Ranks for class `c`
    return np.take(all_rank_indices, rank_indices_label)[0]


def mauc_numpy(labels, probabilities):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        labels (numpy array): A numpy array of the labels
        probabilities (numpy array): A numpy array of the class probabilities
        in the form (m = # data instances):
             [[p(x1c1), p(x1c2), ... p(x1cn)],
              [p(x2c1), p(x2c2), ... p(x2cn)]
                             ...
              [p(xmc1), p(xmc2), ... (pxmcn)]
             ]
    Returns:
        The MAUC as a floating point value.
    """
    num_classes = probabilities.shape[1]

    # Compute A values for all all pairwise comparisons of labels
    # Diagonal is zero and matrix is symmetric
    average_a_values = [[
        separability(labels, probabilities, c1=i, c2=j)
        for i in range(num_classes)
    ] for j in range(num_classes)]
    return np.sum(average_a_values) / float(num_classes *
                                            (num_classes - 1))


@test_utils.run_all_in_graph_and_eager_modes
class SparsemaxTest(tf.test.TestCase):
    def test_multiclass_auc_against_numpy(self):
        pass

    def test_multiclass_auc_against_standard_auc(self):
        pass


if __name__ == '__main__':
    tf.test.main()
