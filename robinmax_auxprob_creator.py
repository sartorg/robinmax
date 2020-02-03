"""Creates the lower-bounding problem

This module contains the definition of the problem that, given the
values of variables y, computes a lower bound for the original
robust influence maximization, using Cplex.

"""

import cplex

def create_lowerbounding_problem(graph, num_seeds, thresh_budget,
                                 max_thresh_dev, weight_budget,
                                 max_weight_dev, epsilon):
    """Create MIP model for Branch-and-Bound search.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        The graph.

    num_seeds : int
        Number of initial nodes to activate.

    thresh_budget : float
        Maximum threshold budget available to the opponent.

    max_thresh_dev : float
        Maximum node threshold deviation.

    weight_budget : float
        Maximum weight budget available to the opponent.

    max_weight_dev : float
        Maximum arc weight deviation.

    epsilon : float
        An small number used to model strict inequalities.

    Returns
    -------
    cplex.Cplex
        A Cplex problem object.
    """
    prob = cplex.Cplex()

    prob.parameters.advance.set(0)

    big_M = [sum(graph.arc_weight_in[i]) + 1 for i in range(graph.num_nodes)]

    # Variables

    # x variables
    x_start = 0
    prob.variables.add(
        obj=[1] * graph.num_nodes,
        types=[prob.variables.type.binary] * graph.num_nodes,
        names=['x_' + str(i) for i in range(graph.num_nodes)])
    # theta variables
    theta_start = x_start + graph.num_nodes
    prob.variables.add(
        lb=[0] * graph.num_nodes,
        ub=[graph.node_threshold[i]*max_thresh_dev
            for i in range(graph.num_nodes)],
        types=[prob.variables.type.continuous] * graph.num_nodes,
        names=['theta_' + str(i) for i in range(graph.num_nodes)])
    # phi variables
    phi_start = theta_start + graph.num_nodes
    phi_pos = dict()
    ind = phi_start
    for i in range(graph.num_nodes):
        for j in range(len(graph.instar[i])):
            phi_pos[i,j] = ind
            ind += 1
    prob.variables.add(
        lb=[0] * graph.num_arcs,
        ub=[graph.arc_weight_in[i][j]*max_weight_dev
            for i in range(graph.num_nodes) for j in range(len(graph.instar[i]))],
        types=[prob.variables.type.continuous] * graph.num_arcs,
        names=['phi_' + str(i) + '_' + str(graph.instar[i][j]) 
               for i in range(graph.num_nodes) for j in range(len(graph.instar[i]))])

    # Constraints
    # \sum_{j \in instar(i)} w_ij*x_j - \sum_{j \in instar(i)} phi_ij <= t_j + \theta_j - \epsilon + M*x_j
    for i in range(graph.num_nodes):
        if not graph.instar[i]:
            continue
        x_ind = [graph.instar[i][j] + x_start for j in range(len(graph.instar[i]))]
        x_val = [graph.arc_weight_in[i][j] for j in range(len(graph.instar[i]))]
        phi_ind = [phi_pos[i,j] for j in range(len(graph.instar[i]))]
        phi_val = [-1 for _ in range(len(phi_ind))]
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=x_ind + [theta_start + i] +
                                       [x_start + i] + phi_ind,
                                       val=x_val + [-1] + [-big_M[i]] + 
                                       phi_val)],
            senses=['L'], rhs=[graph.node_threshold[i] - epsilon[i]],
            names=['x_activation_' + str(i)]
        )
    # \sum_{i \in nodes} \theta_i = B_N
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[i + theta_start for i in range(graph.num_nodes)],
                                   val=[1] * graph.num_nodes)],
        senses=['L'], rhs=[thresh_budget],
        names=['thresh_budget'])
    # \sum_{i \in nodes} \phi_i = B_A
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[i + phi_start for i in range(graph.num_arcs)],
                                   val=[1] * graph.num_arcs)],
        senses=['L'], rhs=[weight_budget],
        names=['weight_budget'])

    # \phi_i_j \leq  max_weight-dev * w_i_j * x_i
    for i in range(graph.num_nodes):
        for j in range(len(graph.instar[i])):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[phi_pos[i, j]] + [graph.instar[i][j]],
                                           val=[1] + [-graph.arc_weight_in[i][j]*max_weight_dev])],
                senses=['L'], rhs=[0],
                names=['phi_activation_' + str(i) + '_' + str(graph.instar[i][j])])

    # This is a maximization problem
    prob.objective.set_sense(prob.objective.sense.minimize)

    return prob

# -- end function

def create_bound_tight_prob(graph, cover_set, cover_index, cover_size,
                            num_covers):
    """Create MIP model for Branch-and-Bound search.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        The graph.

    cover_set : List[List[int]]
        List of activation covers for each node.

    cover_index : List[List[int]]
        Unique index for each cover.

    cover_size : List[int]
        Size of each cover.

    num_covers : int
        Total number of activation covers.

    Returns
    -------
    List[float]
        Upper bound on mu for each node.
    """
    prob = cplex.Cplex()
    prob.set_problem_type(prob.problem_type.LP)

    # Add pi variables
    pi_start = 0
    prob.variables.add(
        lb = [-graph.num_nodes] * num_covers,
        ub = [0] * num_covers,
        names = ['pi_' + str(i) for i in range(num_covers)])
    # Add mu variables
    mu_start = pi_start + num_covers
    prob.variables.add(
        lb = [0] * graph.num_nodes, ub = [graph.num_nodes] * graph.num_nodes,
        names = ['mu_' + str(i) for i in range(graph.num_nodes)])
    # Constraints. Activation for each node.
    for i in range(graph.num_nodes):
        # The pi variables with coefficient +1 are those for covers
        # that node i can activate.
        pos_coeff = sorted(pi_start + cover_index[j][k]
                           for j in graph.outstar[i]
                           for k in range(len(cover_set[j]))
                           if i in cover_set[j][k])
        # The pi variables with coefficient -1 are those for covers
        # that activate node i.
        neg_coeff = [pi_start + cover_index[i][k]
                     for k in range(len(cover_set[i]))]
        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = pos_coeff + neg_coeff +
                                         [mu_start + i],
                                         val = [1] * len(pos_coeff) +
                                         [-1] * len(neg_coeff) + [1])],
            senses = ['L'], rhs = [1], names = ['cover_dual_' + str(i)])
    # Constraint on objective function
    prob.linear_constraints.add(
        lin_expr = [cplex.SparsePair(ind = [i for i in range(mu_start +
                                                             graph.num_nodes)],
                                     val = [size - 1 for size in cover_size] +
                                     [1] * graph.num_nodes)],
        senses = ['L'], rhs = [graph.num_nodes], names =['obj_bound'])
    # This is a maximization problem
    prob.objective.set_sense(prob.objective.sense.maximize)
    return prob
# -- end function

