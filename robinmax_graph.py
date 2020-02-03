"""Read graph data and create corresponding data structures.

This module reads a graph from file, and creates a standardized data
structure that can be used to create Cplex models.

"""

import os
import numpy as np


class LinearThresholdGraph:
    """A graph with thresholds on the nodes and weights on arcs.

    Attributes
    ----------
    num_nodes : int
        The number of nodes.

    num_arcs : int
        The number of arcs.

    outstar : List[List[int]]
        The outstar of each node, given as a list of indices of the
        destination node for each arc leaving a given node.

    instar : List[List[int]]
        The instar of each node, given as a list of indices of the
        source node for each arc entering a given node.

    arc_weight_out : List[List[float]]
        Weight on the arc in the corresponding position in outstar.

    arc_weight_in : List[List[float]]
        Weight on the arc in the corresponding position in instar.

    node_threshold : List[float]
        Activation threshold of a node, indexed from 0 to num_nodes - 1.

    Parameters
    ----------
    threshold : List[float]
        List of node thresholds.

    arc : List[(int, int, float)]
        List of arcs, given as (source, destination, weight).

    name : str
        The name of the graph (or the file it was read from).
    """
    def __init__(self, threshold, arc, name='no_name'):
        """Constructor.
        """
        self.name = name
        self.num_nodes = len(threshold)
        self.num_arcs = len(arc)
        outstar = [list() for i in range(self.num_nodes)]
        instar = [list() for i in range(self.num_nodes)]
        arc_weight_out = [list() for i in range(self.num_nodes)]
        arc_weight_in = [list() for i in range(self.num_nodes)]
        self.node_threshold = list(threshold)
        for (i, j, w) in arc:
            outstar[i-1].append(j-1)
            instar[j-1].append(i-1)
            arc_weight_out[i-1].append(w)
            arc_weight_in[j-1].append(w)
        self.outstar = outstar
        self.instar = instar
        self.arc_weight_out = arc_weight_out
        self.arc_weight_in = arc_weight_in
    # -- end function

    def __str__(self):
        """Convert to string for printing purposes.
        """
        out = 'Number of nodes: {:d}\n'.format(self.num_nodes)
        out += 'Number of arcs: {:d}\n'.format(self.num_arcs)
        out += 'Arcs:\n'
        for (i, star) in enumerate(self.outstar):
            for (index, j) in enumerate(star):
                out += '{:d} {:d} {:f}\n'.format(
                    i + 1, j + 1, self.arc_weight_out[i][index])
        out += 'Nodes:\n'
        for (i, thresh) in enumerate(self.node_threshold):
            out += '{:d} {:f}\n'.format(i + 1, thresh)
        return out
# -- end class
        

def read_text_graph_old(node_file, arc_file):
    """Read graph from files.

    Read a graph from files. The format of the file is the following:
    the node_file contains one
    line per node, with format (label_node, activation_threshold).
    the arc_file contains one line per edge, with format
    (label_source, label_end, weight); 
    Node labels in both files are supposed to match. It is assumed
    that node labels start from 1.

    Parameters
    ----------
    arc_file : string
        Name of the file containing graph connectivity information.

    node_file : string
        Name of the file containing the node activation thresholds.

    Returns
    -------
    LinearThresholdGraph
        Data structure containing the graph.

    """
    arc = list()
    with open(arc_file, 'r') as f:
        for line in f:
            i, j, w = line.split()
            try:
                int_i = int(i)
                assert (int_i >= 1)
                int_j = int(j)
                assert (int_j >= 1)
                float_w = float(w)
                assert (0 <= float_w <= 1)
                arc.append((int_i, int_j, float_w))
            except ValueError:
                print('Indices should be integers (e.g., 35),\n'
                      'weights should be floats (e.g, 0.4).')
    threshold = list()
    index = list()
    with open(node_file, 'r') as f:
        for line in f:
            i, w = line.split()
            try:
                int_i = int(i)
                assert (int_i >= 1)
                float_w = float(w)
                assert (0 <= float_w <= 1)
                index.append(int_i)
                threshold.append(float_w)
            except ValueError:
                print('Indices should be integers (e.g., 35),\n'
                      'weights should be floats (e.g, 0.4).')
    sorted_indices = np.argsort(index)
    sorted_threshold = [threshold[index] for index in sorted_indices]
    graph = LinearThresholdGraph(sorted_threshold, arc)
    return graph        
# -- end function


def read_text_graph(graph_file):
    """Read graph from a file.

    Read a graph from a file.
    It is assumed that node labels start from 1.
    The format of the file is the following:

    NODES
    (index threshold)
    ARCS
    (node_from node_to weight)

    Parameters
    ----------
    graph_file : string
        Name of the file containing graph connectivity information.

    Returns
    -------
    LinearThresholdGraph
        Data structure containing the graph.

    """
    arc = list()
    threshold = list()
    index = list()
    with open(graph_file, 'r') as f:
        # Skip the first line
        next(f)
        is_arc = False
        for line in f:
            if (line[:4] == "ARCS"):
                is_arc = True
                continue
            if is_arc:
                i, j, w = line.split()
                try:
                    int_i = int(i)
                    assert (int_i >= 1)
                    int_j = int(j)
                    assert (int_j >= 1)
                    float_w = float(w)
                    arc.append((int_i, int_j, float_w))
                except ValueError:
                    print('Indices should be integers (e.g., 35),\n')
            else:
                i, w = line.split()
                try:
                    int_i = int(i)
                    assert (int_i >= 1)
                    float_w = float(w)
                    index.append(int_i)
                    threshold.append(float_w)
                except ValueError:
                    print('Indices should be integers (e.g., 35),\n')

    sorted_indices = np.argsort(index)
    sorted_threshold = [threshold[index] for index in sorted_indices]
    graph_name = os.path.basename(graph_file)
    graph = LinearThresholdGraph(sorted_threshold, arc, graph_name)
    return graph
# -- end function

def write_text_graph(graph, graph_file):
    """Write graph to a file.

    Write a graph to a file.
    It is assumed that node labels start from 1.
    The format of the file is the following:

    NODES
    (index threshold)
    ARCS
    (node_from node_to weight)

    Parameters
    ----------

    graph : LinearThresholdGraph
        Data structure containing the graph.

    graph_file : string
        Name of the file where to save the graph.

    """
    with open(graph_file, 'w') as f:
        f.write("NODES\n")
        for i in range(graph.num_nodes):
            f.write("{:d} {:.2f}\n".format(i+1, graph.node_threshold[i]))
        f.write("ARCS\n")
        for i in range(graph.num_nodes):
            for j in range(len(graph.outstar[i])):
                f.write("{:d} {:d} {:.2f}\n".format(i+1, graph.outstar[i][j]+1, graph.arc_weight_out[i][j]))

# -- end function
