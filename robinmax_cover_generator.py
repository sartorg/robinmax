"""Generate minimal activation covers.

This module contains all functions relative to the generation of
activation covers for a linear threshold graph.

"""

import numpy as np
from itertools import chain, combinations


def powerset(indices):
    """Generate power set of a given list of indices

    Parameters
    ----------
    indices : List[int]
        The list of indices for which we should generate the powerset.

    Returns
    -------
    iterator
        An iterator over the powerset, where each element is given as
        a tuple.

    """
    return chain.from_iterable(combinations(indices, n) 
                               for n in range(1, len(indices)+1))
# -- end function


def generate_minimal_covers_brute_force(graph):
    """Generate minimal activation covers.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        Graph for which activation covers should be generated.

    Returns
    -------
    List[List[List[int]]]
        List of activation covers for each node.
    """
    list_covers = [list() for i in range(graph.num_nodes)]
    for i in range(graph.num_nodes):
        size = len(graph.instar[i])
        # Generate all combinations of indices from 0 to size-1
        for cover in chain.from_iterable(combinations(range(size), n) 
                                         for n in range(1, size + 1)):
            if (sum(graph.arc_weight_in[i][j] for j in cover) >= 
                graph.node_threshold[i]):
                # This is a cover. Check minimality
                minimal = True
                for k in cover:
                    if (sum(graph.arc_weight_in[i][k] for j in cover 
                            if j != k) >= graph.node_threshold[i]):
                        minimal = False
                        break
                if (minimal):
                    list_covers[i].append(sorted(graph.instar[i][j]
                                                 for j in cover))
                # -- end if
            # -- end if
        # -- end for
    # -- end for
    return list_covers
# -- end function


def generate_covers_with_thresh(graph, max_cover_size, thresh_budget, 
                                max_thresh_dev, weight_budget, max_weight_dev):
    """Generate activation covers. All covers up to max_cover_size + 1
    will be generated, but only those up to max_cover_size will be
    returned. An additional list contains a the thresholds for
    minimality and validity.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        Graph for which activation covers should be generated.

    max_cover_size : int
        Maximum size of activation covers to consider.

    thresh_budget : float
        Maximum threshold budget available to the opponent.

    max_thresh_dev : float
        Maximum node threshold deviation.

    weight_budget : float
        Maximum weight budget available to the opponent.

    max_weight_dev : float
        Maximum arc weight deviation.

    Returns
    -------
    List[List[List[int]]]
        List of activation covers for each node.

    List[List[(float, float)]]
        The thresholds (min_minimality, max_validity) for each cover.

    """

    covers = [list() for _ in range(graph.num_nodes)]

    # Generate covers of a slightly larger size.
    covers, thresholds = generate_minimal_covers(
        graph, max_cover_size, thresh_budget, 
        max_thresh_dev, weight_budget, max_weight_dev)
    return covers, thresholds


def generate_minimal_covers(graph, max_cover_size, thresh_budget, 
                            max_thresh_dev, weight_budget, max_weight_dev):
    """Generate minimal activation covers.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        Graph for which activation covers should be generated.

    max_cover_size : int
        Maximum size of activation covers to consider. Any activation
        cover larger than this size will not be generated.
    
    thresh_budget : float
        Maximum threshold budget available to the opponent.

    max_thresh_dev : float
        Maximum node threshold deviation.

    weight_budget : float
        Maximum weight budget available to the opponent.

    max_weight_dev : float
        Maximum arc weight deviation.

    Returns
    -------
    List[List[List[int]]]
        List of activation covers for each node.

    """
    list_covers = [list() for i in range(graph.num_nodes)]
    thresholds = [list() for i in range(graph.num_nodes)]
    for i in range(graph.num_nodes):
        # Sort the arcs by decreasing weight
        sort_order = [i for i in
                      reversed(np.argsort(graph.arc_weight_in[i]))]
        covers = []

        # Generate covers recursively
        recursive_generate_covers(graph.arc_weight_in[i], sort_order,
                                  graph.node_threshold[i], [], 0,
                                  max_cover_size, thresh_budget, max_thresh_dev,
                                  covers, weight_budget, max_weight_dev)
        # Translate the arc indices in the sorted list, to the index
        # of the node that these arcs originate from
        list_covers[i] = [sorted(graph.instar[i][sort_order[j]]
                                 for j in cover) for cover in covers]
        # Skip if we did not find any cover
        if len(covers) == 1 and len(covers[0]) == 0:
            continue
        for cover in covers:
            value = sum(graph.arc_weight_in[i][sort_order[j]] for j in cover)
            minimality = value - min(graph.arc_weight_in[i][sort_order[j]] for j in cover)
            thresholds[i].append((minimality, value))
    return list_covers, thresholds
# -- end function


def recursive_generate_covers(weights, sort_order, threshold, curr_set,
                              curr_index, max_cover_size, thresh_budget,
                              max_thresh_dev, all_sets, weight_budget,
                              max_weight_dev):
    """Generate all minimal activation covers for given node.

    Recursive function to generate all minimal activation covers, in a
    dynamic programming fashion.

    Parameters
    ----------
    weights : List[float]
        List of arc weights.

    sort_order : List[int]
        Indices of weights sorted by decreasing value.

    threshold : float
        Activation threshold for the node.

    curr_set : List[int]
        Current set of arcs in the cover.

    curr_index : int
        Position of arc weights yet to be considered.

    max_cover_size : int
        Maximum size of covers to consider.

    thresh_budget : float
        Maximum threshold budget available to the opponent.

    max_thresh_dev : float
        Maximum node threshold deviation.

    all_sets : List[List[int]]
        The list of all covers discovered.
    
    weight_budget : float
        Maximum weight budget available to the opponent.

    max_weight_dev : float
        Maximum arc weight deviation.

    Returns
    -------
    bool
        True if a cover was generated, False otherwise

    """
    if (curr_index > len(weights) or len(curr_set) > max_cover_size):
        return False
    any_generated = False
    if (sum(weights[sort_order[i]] for i in curr_set) >= threshold):
        all_sets.append([i for i in curr_set])
        any_generated = True
        max_curr_set_dev = sum(weights[sort_order[i]] * max_weight_dev for i in curr_set)
        if (sum(weights[sort_order[i]] for i in curr_set) >=
            threshold + min(thresh_budget, threshold * max_thresh_dev) + min(weight_budget, max_curr_set_dev)):
            return True
    for j in range(curr_index, len(weights)):
        curr_set.append(j)
        result = recursive_generate_covers(weights, sort_order, threshold,
                                           curr_set, j + 1, max_cover_size, thresh_budget,
                                           max_thresh_dev, all_sets, weight_budget,
                                           max_weight_dev)
        curr_set.pop()
        if not result:
            break
        else:
            any_generated = True
    return any_generated
    
    
