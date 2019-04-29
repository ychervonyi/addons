from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.metrics import metric_variable
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops


def multiclass_auc(labels,
                   predictions,
                   name=None,
                   updates_collections=None,
                   metrics_collections=None):
    """Computes multi-class AUC (MAUC) as described in Hand, D.J. & Till, R.J.
    Machine Learning (2001) 45: 171. https://doi.org/10.1023/A:1010920819831.
    [1]

    The `multiclass_auc` function computes a separability matrix (denoted as A
    in the paper) for all pairwise comparisons of  different classes. Each
    value in the separability matrix is an overall measure of how well
    separated are the estimated distributions for the two considered classes.
    The overall performance of the classification rule in separating all
    classes is then the average of the separability matrix (divided by 2
    because the matrix is symmetric).

    USAGE NOTE: this approach requires storing all of the predictions and
    labels for a single evaluation in memory, so it may not be usable when
    the evaluation
    batch size and/or the number of evaluation steps is very large.

    Args:
        labels: A `Tensor` with the shape [batch_size, num_points].
        predictions: A `Tensor` with shape [batch_size, num_points,
            num_classes].
        name: A string used as the name for this metric.

    Returns:
        A metric for the multi-class AUC.
    """

    with variable_scope.variable_scope(name, 'mauc', (labels, predictions)):
        # If not tensors - convert
        labels = ops.convert_to_tensor(labels)
        predictions = ops.convert_to_tensor(predictions)

        # Extract number of classes
        shapes = predictions.get_shape().as_list()
        num_classes = shapes[-1]

        # If batch - flatten
        labels = array_ops.reshape(labels, [-1])
        predictions = array_ops.reshape(predictions, [-1, num_classes])

        # Accumulate predictions and labels
        preds_accum, update_preds = streaming_concat(
            predictions, name='concat_preds')
        labels_accum, update_labels = streaming_concat(
            labels, name='concat_labels')
        update_op = control_flow_ops.group(update_preds, update_labels)

        def _compute_mauc(predictions, labels):
            """Computes multi-class auc."""
            # Extract number of classes
            shapes = predictions.get_shape().as_list()
            num_classes = shapes[-1]
            a_mtr = ops.convert_to_tensor([[
                _separability(labels, predictions, c1=i, c2=j)
                for i in range(num_classes)
            ] for j in range(num_classes)])
            return math_ops.divide(
                math_ops.reduce_sum(a_mtr), num_classes * (num_classes - 1.0))

        mauc_value = _compute_mauc(
            predictions=preds_accum, labels=labels_accum)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        if metrics_collections:
            ops.add_to_collections(metrics_collections, mauc_value)
        return mauc_value, update_op


def _separability(labels, predictions, c1=0, c2=1):
    """Computes the measure of separability between classes labeled as `c1` and
    `c2`. This quantity is denoted as \hat{A} in the paper. It is defined as
    the probability that a randomly drawn member of class `c2` will have a
    lower estimated probability of belonging to class `c1` than a randomly
    drawn member of class `c1`

    Args:
        labels: A `Tensor` of the same shape as `predictions`.
        predictions: A `Tensor` with shape [batch_size, num_classes].
        c1: Label for class 1.
        c2: Label for class 2.

    Returns:
        A `Tensor` as the measure of separability between classes labeled as
            `c1` and `c2`. The tensor shape is [batch_size, 1].
    """

    # Get a list of indices for the classes that are being compared (`c1` and
    # `c2`)
    indices_selected_classes = array_ops.where(
        gen_math_ops.logical_or(
            math_ops.equal(labels, c1), math_ops.equal(labels, c2)))
    # Use the indices to select corresponding labels and predictions
    labels_selected_classes = array_ops.gather(labels,
                                               indices_selected_classes)
    preds_selected_classes = array_ops.squeeze(
        array_ops.gather(predictions, indices_selected_classes), 1)
    # Concatenate labels and corresponding predictions
    points_selected_classes = array_ops.concat([
        preds_selected_classes,
        math_ops.cast(labels_selected_classes, preds_selected_classes.dtype)
    ], 1)

    # Compute rank for each label - simply 1 based array of ints
    all_rank_indices = math_ops.range(
        1,
        array_ops.shape(labels_selected_classes)[0] + 1)

    # Number of class 1 and class 2 instances
    n1 = _count(labels_selected_classes, c1)
    n2 = _count(labels_selected_classes, c2)

    # Compute total ranks for class 1 and class 2
    sum_ranks_c1 = _total_rank_for_class(points_selected_classes, c1,
                                         all_rank_indices)
    sum_ranks_c2 = _total_rank_for_class(points_selected_classes, c2,
                                         all_rank_indices)

    # Compute separability (denoted as A in the paper [1] - eqn (3) with 1->2
    # and 0->1 and averaged over swapped classes).
    # Note that if a batch contains no examples of one of the classes A value
    # diverges due to division by 0, so `safe_div` is used to ignores these
    # cases
    return math_ops.div_no_nan(
        sum_ranks_c1 + sum_ranks_c2 - _sum_first_n(n1) - _sum_first_n(n2),
        2.0 * n1 * n2)


def _count(x, value):
    """ Count occurrences of `value` in `Tensor` x.
    Args:
       x:  A `Tensor` in which occurrences of `value` element will be counted.
       value: A value of the element whose occurrence will be counted.

    Returns:
        A `Tensor` as the number of occurrences of `value` in `x`.
    """
    return math_ops.reduce_sum(
        math_ops.cast(math_ops.equal(x, value), dtypes.float32))


def _sum_first_n(N):
    """Computes sum of the first N positive integers.

    sum = N (N + 1)/2

    Args:
        N: A `Tensor` as the integer N.

    Returns:
        A `Tensor` as the sum of the first N positive integers.
    """
    return math_ops.divide(N * (N + 1.0), 2.0)


def _total_rank_for_class(points, c, all_rank_indices):
    """Computes total rank for class labeled as `c`.

    Args:
        points: A `Tensor` of concatenated predictions and labels.
            c: A label of the class by whose probabilities the sorting is done.
        all_rank_indices: A `Tensor` containing all ranks for the classes
            being compared.

    Returns:
        Total rank for class `c` of the batch.
    """

    # Select predictions for class `c`
    preds_class = array_ops.slice(points, [0, c], [-1, 1])

    # Sort `points` by predictions for class `c`
    sorted_indices = sort_ops.argsort(preds_class, 0)
    points_sorted_by_prob_class = array_ops.squeeze(
        array_ops.gather(points, sorted_indices), 1)

    # Now we need only labels - extract them from the last dimension
    last_dim = points.get_shape().as_list()[1] - 1
    labels_sorted_by_prob_class = array_ops.slice(points_sorted_by_prob_class,
                                                  [0, last_dim], [-1, 1])

    # Convert labels back to int
    labels_sorted_by_prob_class_int = math_ops.cast(
        labels_sorted_by_prob_class, dtypes.int64)

    # Labels of class `c`
    rank_indices_class = array_ops.where(
        math_ops.equal(labels_sorted_by_prob_class_int, c))
    # array_ops.where adds another dimension - ignore it
    rank_indices_class_values = array_ops.slice(rank_indices_class, [0, 0],
                                                [-1, 1])

    # Total rank
    rank_elements_class = array_ops.gather(all_rank_indices,
                                           rank_indices_class_values)
    return math_ops.cast(
        math_ops.reduce_sum(rank_elements_class), dtypes.float32)


# TODO: The following function is copied from tf.contrib. Is there a better way
#  to use it? maybe import?
def streaming_concat(values,
                     axis=0,
                     max_size=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
    """Concatenate values along an axis across batches.

    The function `streaming_concat` creates two local variables, `array` and
    `size`, that are used to store concatenated values. Internally, `array` is
    used as storage for a dynamic array (if `maxsize` is `None`), which ensures
    that updates can be run in amortized constant time.

    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that appends the values of a tensor and returns the
    length of the concatenated axis.

    This op allows for evaluating metrics that cannot be updated incrementally
    using the same framework as other streaming metrics.

    Args:
        values: `Tensor` to concatenate. Rank and the shape along all axes
            other than the axis to concatenate along must be statically known.
        axis: optional integer axis to concatenate along.
        max_size: optional integer maximum size of `value` along the given
            axis. Once the maximum size is reached, further updates are no-ops.
            By default, there is no maximum size: the array is resized as
            necessary.
        metrics_collections: An optional list of collections that `value`
            should be added to.
        updates_collections: An optional list of collections `update_op` should
            be added to.
        name: An optional variable_scope name.

    Returns:
        value: A `Tensor` representing the concatenated values.
        update_op: An operation that concatenates the next values.

    Raises:
        ValueError: if `values` does not have a statically known rank, `axis`
            is not in the valid range or the size of `values` is not
            statically known along any axis other than `axis`.
    """
    with variable_scope.variable_scope(name, 'streaming_concat', (values,)):
        # pylint: disable=invalid-slice-index
        values_shape = values.get_shape()
        if values_shape.dims is None:
            raise ValueError('`values` must have known statically known rank')

        ndim = len(values_shape)
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise ValueError('axis = %r not in [0, %r)' % (axis, ndim))

        fixed_shape = [
            dim.value for n, dim in enumerate(values_shape) if n != axis
        ]
        if any(value is None for value in fixed_shape):
            raise ValueError(
                'all dimensions of `values` other than the dimension to '
                'concatenate along must have statically known size')

        # We move `axis` to the front of the internal array so assign ops can be
        # applied to contiguous slices
        init_size = 0 if max_size is None else max_size
        init_shape = [init_size] + fixed_shape
        array = metric_variable(
            init_shape, values.dtype, validate_shape=False, name='array')
        size = metric_variable([], dtypes.int32, name='size')

        perm = [
            0 if n == axis else n + 1 if n < axis else n for n in range(ndim)
        ]
        # TODO: sanity-check says "Value 'array' is unsubscriptable"
        valid_array = array[:size]
        valid_array.set_shape([None] + fixed_shape)
        value = array_ops.transpose(valid_array, perm, name='concat')

        values_size = array_ops.shape(values)[axis]
        if max_size is None:
            batch_size = values_size
        else:
            batch_size = math_ops.minimum(values_size, max_size - size)

        perm = [axis] + [n for n in range(ndim) if n != axis]
        batch_values = array_ops.transpose(values, perm)[:batch_size]

        def reallocate():
            next_size = _next_array_size(new_size)
            next_shape = array_ops.stack([next_size] + fixed_shape)
            new_value = array_ops.zeros(next_shape, dtype=values.dtype)
            old_value = array.value()
            assign_op = state_ops.assign(
                array, new_value, validate_shape=False)
            with ops.control_dependencies([assign_op]):
                copy_op = array[:size].assign(old_value[:size])
            # return value needs to be the same dtype as no_op() for cond
            with ops.control_dependencies([copy_op]):
                return control_flow_ops.no_op()

        new_size = size + batch_size
        array_size = array_ops.shape_internal(array, optimize=False)[0]
        maybe_reallocate_op = control_flow_ops.cond(
            new_size > array_size, reallocate, control_flow_ops.no_op)
        with ops.control_dependencies([maybe_reallocate_op]):
            append_values_op = array[size:new_size].assign(batch_values)
        with ops.control_dependencies([append_values_op]):
            update_op = size.assign(new_size)

        if metrics_collections:
            ops.add_to_collections(metrics_collections, value)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return value, update_op
        # pylint: enable=invalid-slice-index


def _next_array_size(required_size, growth_factor=1.5):
    """Calculate the next size for reallocating a dynamic array.

    Args:
        required_size: number or tf.Tensor specifying required array capacity.
        growth_factor: optional number or tf.Tensor specifying the growth
            factor between subsequent allocations.

    Returns:
        tf.Tensor with dtype=int32 giving the next array size.
    """
    exponent = math_ops.ceil(
        math_ops.log(math_ops.cast(required_size, dtypes.float32)) /
        math_ops.log(math_ops.cast(growth_factor, dtypes.float32)))
    return math_ops.cast(math_ops.ceil(growth_factor**exponent), dtypes.int32)
