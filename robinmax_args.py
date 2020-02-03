"""Defines the parameters of the algorithm.

This module contains the defintion of the class that
handles all parameters of the algorithm.

"""

import robinmax_utils as utils

class RobinmaxArgs:
    """ The parameter class. Contains all the 
    parameters necessary for the algorithm.

    """
    def __init__(self, graph, thresh_budget, max_thresh_dev,
                 weight_budget, max_weight_dev, max_cover_size,
                 time_limit, num_seeds, heuristics, disable_cuts,
                 solve_as_lp, debug_level, out_f):
        """Constructor.
        """
        # Performs some sanity checks
        if heuristics not in [-1, 1, 2]:
            print('Invalid value for heuristics parameter.' + 
            'Check python3 robinmax.py --help.')

        # Compute the epsilon to use throughout the algorithm
        epsilon = utils.epsilon(graph)
        
        self.graph = graph
        self.thresh_budget = thresh_budget
        self.max_thresh_dev = max_thresh_dev
        self.weight_budget = weight_budget
        self.max_weight_dev = max_weight_dev
        self.max_cover_size = max_cover_size
        self.time_limit = time_limit
        self.num_seeds = num_seeds
        self.heuristics = heuristics
        self.disable_cuts = disable_cuts
        self.solve_as_lp = solve_as_lp
        self.epsilon = epsilon
        self.debug_level = debug_level
        self.out_f = out_f

        # Set to 0 the max deviations if their coresponding
        # budget is 0.
        if (self.thresh_budget == 0 and self.max_thresh_dev > 0):
            print('******* Warning: threshold budget is zero but its max deviation is not.')
            print('Setting max threshold deviation to 0.')
            self.max_thresh_dev = 0
        if (self.weight_budget == 0 and self.max_weight_dev > 0):
            print('******* Warning: weight budget is zero but its max deviation is not.')
            print('--> Setting max weight deviation to 0.')
            self.max_weight_dev = 0

        # Setting default value of max_cover_size
        if (self.max_cover_size == -1):
            self.max_cover_size = self.graph.num_nodes

    def __str__(self):
        """Convert to string for printing purposes.
        """
        out = 'GRAPH\n'
        out += 'Name: {:s}\n'.format(self.graph.name)
        out += 'Nodes: {:d}\n'.format(self.graph.num_nodes)
        out += 'Arcs: {:d}\n'.format(self.graph.num_arcs)
        out += '\n'
        out += 'PARAMETERS\n'
        out += 'Seeds: {:d}\n'.format(int(self.num_seeds))
        out += 'Cover size: {:d}\n'.format(int(self.max_cover_size))
        out += 'Robustness threshold budget: {:.2f}\n'.format(self.thresh_budget)
        out += 'Max threshold deviation: {:.2f}\n'.format(self.max_thresh_dev)
        out += 'Robustness weight budget: {:.2f}\n'.format(self.weight_budget)
        out += 'Max weight deviation: {:.2f}\n'.format(self.max_weight_dev)
        out += 'Time limit: {:.1f}\n'.format(self.time_limit)
        out += 'Disable cuts: {:s}\n'.format(str(self.disable_cuts))
        out += 'Solve as LP: {:s}\n'.format(str(self.solve_as_lp))
        out += 'Epsilon: {:.2e}\n'.format(self.epsilon)
        out += 'Debug level: {:s}\n'.format(str(self.debug_level))
        out += 'Output file: {:s}\n'.format(str(self.out_f.name))
        out += '\n'
        
        return out
