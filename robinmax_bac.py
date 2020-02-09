"""Branch-and-Cut for robust influence maximization.

This module contains the definition of the Branch-and-Cut problem for
robust influence maximization, using Cplex.

"""

import sys
import os
import time
import argparse
import ast
import numpy as np
import collections

import cplex
from cplex.exceptions import CplexError
import robinmax_graph as gr
import robinmax_cover_generator as cg
import robinmax_callbacks as cb
import robinmax_auxprob_creator as aux
import robinmax_column_generation as col
import robinmax_pricing as pr

def bac_restart(graph, num_seeds, max_cover_size,
                thresh_budget=0, max_thresh_dev=0.0,
                weight_budget=0.0, max_weight_dev=0.0,
                max_time=3600, epsilon=1e-6, debugging=False,
                disable_cuts=False, lp=False, covers=[],
                thresholds=[], save_pruned=False,
                run_as_heuristic=False, cg_init_iters=20,
                max_columns_per_round=10000,
                max_pricing_iters=0, max_col_iters_per_round=0,
                out_f=sys.__stdout__):
    """Create MIP model for Branch-and-Bound search.

    Parameters
    ----------
    graph : `robinmax_graph_reader.LinearThresholdGraph`
        The graph.

    num_seeds : int
        Number of initial nodes to activate.

    max_cover_size : int
        Maximum size of the activation covers.

    thresh_budget : float
        Maximum threshold budget available to the opponent.

    max_thresh_dev : float
        Maximum node threshold deviation.

    weight_budget : float
        Maximum weight budget available to the opponent.

    max_weight_dev : float
        Maximum arc weight deviation.

    max_time : float
        Maximum time

    epsilon : float
        An small number used to model strict inequalities.

    laxy_relax_factor : float
        Coefficient used to relax lazy cuts at different incumbents.

    debugging : bool
        If True, additional info are printed in the standard output.

    disable_cuts: bool
        If True, disable all CPLEX cuts.

    lp: bool
        If True, solve the relaxed problem.

    covers: List[List[int]]
        List of activation covers for each node.
        Note: during tests, it may be convenient to generate the covers
        only once outside this function.

    thresholds: List[List[(float, float)]]
        List of thresholds (minimality, validity) for each cover.

    save_pruned: bool
        Saved information on pruned nodes (slow).

    run_as_heuristic: bool
        Runs with column generation

    cg_init_iters: int
        Number of iterations in initialization of column generation

    max_columns_per_round: int
        Maximum columns per column generation round.

    pricing_max_iters: int
        Maximum number of iterations of the pricing problem.

    max_col_iters_per_round: int
        Maximum number of iterations allowed within the column 
        generation heuristic inner loop.

    out_f : file object
        The file where the standard output should be redirected.

    Returns
    -------
    cplex.Cplex
        A Cplex problem object.

    """

    # Default values
    if (max_pricing_iters == 0):
        max_pricing_iters = graph.num_nodes*0.05

    if (max_col_iters_per_round == 0):
        max_col_iters_per_round = 2 * graph.num_nodes

    
    prob = cplex.Cplex()

    prob.parameters.threads.set(1)
    prob.parameters.mip.display.set(2)
    prob.parameters.mip.limits.treememory.set(4096)
    #prob.parameters.mip.cuts.mircut.set(2)
    prob.parameters.emphasis.numerical.set(0)
    #prob.parameters.mip.tolerances.integrality.set(1e-16)

    if disable_cuts:
        prob.parameters.mip.cuts.mcfcut.set(-1)
        prob.parameters.mip.cuts.localimplied.set(-1)
        prob.parameters.mip.cuts.implied.set(-1)
        prob.parameters.mip.cuts.gubcovers.set(-1)
        prob.parameters.mip.cuts.gomory.set(-1)
        prob.parameters.mip.cuts.pathcut.set(-1)
        prob.parameters.mip.cuts.flowcovers.set(-1)
        prob.parameters.mip.cuts.disjunctive.set(-1)
        prob.parameters.mip.cuts.covers.set(-1)
        prob.parameters.mip.cuts.cliques.set(-1)
        prob.parameters.mip.cuts.zerohalfcut.set(-1)
        prob.parameters.mip.cuts.liftproj.set(-1)
        prob.parameters.mip.cuts.mircut.set(-1)

    if not debugging:
        prob.set_log_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

    if (run_as_heuristic):
        prob.parameters.emphasis.mip.set(4)
    
    if len(covers) > 0:
        cover_set = covers
        thresholds = thresholds
    else:
        cover_set, thresholds = cg.generate_minimal_covers(
            graph, max_cover_size, thresh_budget, max_thresh_dev,
            weight_budget, max_weight_dev)

    # We need to give each cover a unique index
    cover_index = [list() for _ in range(graph.num_nodes)]
    # For each cover, we store its indices (i, j) to find the
    # corresponding set in cover_set
    cover_pointer = list()
    cover_size = list()
    num_covers = 0
    for i in range(graph.num_nodes):
        # Look at how many covers are associated with this node, and
        # assign to them the indices from num_covers to num_covers
        # plus the number of covers at this node
        cover_index[i] = [num_covers + j for j in range(len(cover_set[i]))]
        for j in range(len(cover_index[i])):
            cover_pointer.append([i, j])
            cover_size.append(len(cover_set[i][j]))
        num_covers += len(cover_index[i])

    if debugging:
        for i in range(graph.num_nodes):
            print('Node', i, file=out_f)
            print('Covers:', cover_set[i], file=out_f)
            print('Cover indices:', cover_index[i], file=out_f)
            print('Thresholds:', thresholds[i], file=out_f)

    if (debugging):
        print('Generated {:d} covers.\n'.format(num_covers), file=out_f)

    # Add y variables
    if lp:
        prob.variables.add(
            obj=[big_M] * graph.num_nodes,
            lb=[0] * graph.num_nodes,
            ub=[1] * graph.num_nodes,
            types=[prob.variables.type.continuous] * graph.num_nodes,
            names=['y_' + str(i) for i in range(graph.num_nodes)])
    else:
        prob.variables.add(
            types=[prob.variables.type.binary] * graph.num_nodes,
            names=['y_' + str(i) for i in range(graph.num_nodes)])
    
    # Add mu variables
    mu_start = graph.num_nodes
    prob.variables.add(
        obj=[1] * graph.num_nodes,
        lb=[0] * graph.num_nodes,
        ub = [cplex.infinity] * graph.num_nodes,
        types=[prob.variables.type.continuous] * graph.num_nodes,
        names=['mu_' + str(i) for i in range(graph.num_nodes)])

    # Add pi variables
    pi_start = mu_start + graph.num_nodes
    prob.variables.add(
        obj=[size - 1 for size in cover_size],
        lb=[-cplex.infinity] * num_covers,
        ub=[0] * num_covers,
        types=[prob.variables.type.continuous] * num_covers,
        names=['pi_' + str(i) for i in range(num_covers)])
            
    # Constraints. Activation for each node.
    for i in range(graph.num_nodes):
        # The pi variables with coefficient +1 are those for covers
        # that node i can activate.
        pos_ind = sorted(pi_start + cover_index[j][k]
                         for j in graph.outstar[i]
                         for k in range(len(cover_set[j]))
                         if i in cover_set[j][k])
        # The pi variables with coefficient -1 are those for covers
        # that activate node i.
        neg_ind = [pi_start + cover_index[i][k]
                   for k in range(len(cover_set[i]))]

        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=pos_ind + neg_ind +
                                         [mu_start + i],
                                         val=[1] * len(pos_ind) +
                                         [-1] * len(neg_ind) + [1])],
            senses=['L'], rhs=[1], names=['cover_dual_' + str(i)])
    # Constraint on number of seed nodes
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[i for i in range(graph.num_nodes)],
                                   val=[1] * graph.num_nodes)],
        senses=['E'], rhs=[num_seeds],
        names=['seed_size_constr'])

    # Indicator constraints
    for i in range(graph.num_nodes):
        prob.indicator_constraints.add(
            lin_expr=cplex.SparsePair(ind = [mu_start + i],
                                      val = [1]),
            sense='E', rhs=0.0, indvar=i, complemented=1,
            name='indicator_y_' + str(i), indtype=1)

    # This is a maximization problem
    prob.objective.set_sense(prob.objective.sense.maximize)

    is_robust = ((thresh_budget > 0 and max_thresh_dev > 0) or 
        (weight_budget > 0 and max_weight_dev > 0))

    pricing_problems = []
    if (run_as_heuristic):
        pricing_problems = pr.create_pricing_problems(graph, thresh_budget,
            max_thresh_dev, max_weight_dev, debugging)

    prob.parameters.timelimit.set(max_time)
    start = prob.get_time()
    optimal = False
    best_sol_value = -1.0
    nodes_visited = 0
    current_obj = []
    previous_num_covers = -1
    best_obj = -1.0
    iterations = -1
    while (not optimal and prob.get_time() - start < max_time*0.95):
        # Keep a list of the added lazy constraints
        lazy_constraints = list()

        iterations += 1

        if (is_robust or run_as_heuristic):       
            lazy_constr = cb.LazyConstraint(
                graph, num_seeds, cover_set, thresholds, cover_index,
                cover_pointer, cover_size, num_covers,
                thresh_budget, max_thresh_dev, weight_budget, 
                max_weight_dev, lazy_constraints,
                epsilon, run_as_heuristic,
                best_sol_value if (iterations > cg_init_iters) else -1, 
                debugging, out_f)

            contextmask = 0
            contextmask |= cplex.callbacks.Context.id.candidate
            if (save_pruned):
                contextmask |= cplex.callbacks.Context.id.relaxation
            prob.set_callback(lazy_constr, contextmask)

            # if (iterations > cg_init_iters or not run_as_heuristic):
            #     prob.parameters.mip.tolerances.lowercutoff.set(
            #         best_sol_value - 1.0e-8)
            prob.parameters.timelimit.set(
                max(max_time - (prob.get_time() - start), 0))            

        if ((is_robust) and (debugging or not run_as_heuristic)):
            print('{:s}  {:s}   {:s}'.format(str.center('Visited', 9),
                                             str.center('Rejected', 9), 
                                             str.center('Best', 9),
                                             file=out_f))
        prob.parameters.threads.set(1)
        prob.solve()
        #prob.write("last_mip1.lp")

        improving = False
        if ((is_robust or run_as_heuristic) and
            best_sol_value < lazy_constr.best_sol_value):
            best_solution = lazy_constr.best_solution
            best_sol_value = lazy_constr.best_sol_value
            improving = True

        if (debugging or not run_as_heuristic):
            if (prob.solution.get_status() == prob.solution.status.MIP_optimal
                or 
                prob.solution.get_status() == prob.solution.status.optimal):
                print('\nOptimal solution found!\n', file=out_f)
            else:
                print('\nThe algorithm was not able to find the optimal ' +
                      'solution.', file=out_f)
                print('Cplex status: {:s}\n'.format(
                    prob.solution.get_status_string()), file=out_f)

        try:
            nodes_visited += prob.solution.progress.get_num_nodes_processed()
            if (not is_robust and not run_as_heuristic and
                prob.solution.get_objective_value() > best_sol_value):
                best_sol_value = prob.solution.get_objective_value()
                best_solution = prob.solution.get_values(0, graph.num_nodes)

            if (prob.solution.get_status() == prob.solution.status.MIP_optimal
                or 
                prob.solution.get_status() == prob.solution.status.optimal):
                optimal = True
            elif (len(lazy_constr.lazy_constraints) > 0):
                ind, val = lazy_constr.lazy_obj[0]
                prob.objective.set_linear([(int(ind[i]), float(val[i]))
                                           for i in range(len(ind))])
                current_obj = lazy_constr.lazy_obj
                if (debugging):
                    print("New incumbent as obj: ",
                        [i for i in range(graph.num_nodes)
                         if current_obj[1][0][i] >= 0.5])
        except Exception as ex:
            print("Exception: ", ex)
            break

        #prob.write("last_mip2.lp")

        if (run_as_heuristic):
            
            time_left = max(max_time - (prob.get_time() - start), 0)

            print('ITERATION: {:d}'.format(iterations))

            # Generate covers through column generation
            num_generated_covers, obj, all_generated, all_indices = col.column_generation(
                graph, [current_obj[1]], cover_set, thresholds,
                cover_index, cover_pointer, cover_size, num_covers,
                thresh_budget, max_thresh_dev, weight_budget,
                max_weight_dev, time_left, epsilon, current_obj, 
                lazy_constr.best_sol_value, pricing_problems,
                max_pricing_iters, max_col_iters_per_round, True, 
                max_columns_per_round, debugging, out_f)

            if (num_generated_covers > 0):
                optimal = False
                add_columns(prob, graph, cover_pointer, thresholds, all_generated,
                     all_indices, current_obj, debugging)

            if (round(best_obj) < round(obj)):
                best_obj = obj
            
            print('Best objective: {:.2f}'.format(lazy_constr.best_sol_value))
            print('All covers (#): ', num_covers + num_generated_covers)
            print('Generated covers (#): ', num_generated_covers)

            previous_num_covers = num_covers
            num_covers += num_generated_covers

            # Small check that we added all the generated covers
            assert(num_covers == sum([len(cover_set[i]) for i in range(graph.num_nodes)]))
            if (iterations < cg_init_iters):
                # Add no-good cut
                incumbent, theta, phi = current_obj[1]
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i for i in range(graph.num_nodes)
                             if round(incumbent[i]) == 1],
                        val=[1 for i in range(graph.num_nodes)
                             if round(incumbent[i]) == 1])],
                    senses=['L'],
                    rhs=[num_seeds - 1],
                    names=['no_good_' + str(iterations)])
            elif (iterations == cg_init_iters):
                # Remove all no-good cuts
                prob.linear_constraints.delete(
                    ['no_good_' + str(i) for i in range(cg_init_iters)])
                # Set emphasis back to default
                #prob.parameters.emphasis.mip.set(0)
            
            #prob.write("last_mip3.lp")

    time_elapsed = prob.get_time() - start

    results = [time_elapsed, nodes_visited, 100, graph.num_nodes,
               best_sol_value, num_covers, len(lazy_constraints),
               0, 0, 0, 0, 0]

    try:

        lb_prob = aux.create_lowerbounding_problem(
            graph, num_seeds, thresh_budget, max_thresh_dev,
            weight_budget, max_weight_dev, epsilon)

        lb_prob.variables.set_lower_bounds(
            [(i, best_solution[i]) for i in range(graph.num_nodes)])
        # Solve lower bounding problem
        lb_prob.set_results_stream(None)
        lb_prob.set_log_stream(None)
        lb_prob.parameters.threads.set(1)
        lb_prob.solve()
        if (lb_prob.solution.get_status() !=
            lb_prob.solution.status.MIP_optimal):
            raise RuntimeError('Could not solve sub-MIP')
        # Get the optimal solution
        theta = lb_prob.solution.get_values(graph.num_nodes,
                                            2*graph.num_nodes-1)
        phi = lb_prob.solution.get_values(2*graph.num_nodes,
                                          2*graph.num_nodes +
                                          graph.num_arcs - 1)
        lb_prob.end()
        best_objective = best_sol_value
        best_bound = prob.solution.MIP.get_best_objective()        

    except Exception as ex:
        print('Exception raised', ex)
        if debugging:
            prob.write('last_problem.lp')

        

    try:
        mip_gap = prob.solution.MIP.get_mip_relative_gap()
    except:
        mip_gap = (graph.num_nodes - best_sol_value) / graph.num_nodes


    if (debugging or not run_as_heuristic):
        print('')
        print('CPLEX time (s): {:.2f}'.format(time_elapsed), file=out_f)
        print('Nodes (#): {:d}'.format(nodes_visited), file=out_f)
        print('Gap (%): {:.2f}'.format(mip_gap * 100), file=out_f)
        print('Best bound: {:.2f}'.format(best_bound), file=out_f)
        print('Best objective: {:.2f}'.format(best_objective), file=out_f)
        print('Covers (#): {:d}'.format(num_covers), file=out_f)
        print('Generated lazy cuts (#): {:d}'.format(len(lazy_constraints)), file=out_f)
        print('Nonzero theta at optimum (#): {:d}'.format(np.count_nonzero(theta)), file=out_f)
        print('Max theta at optimum: {:.4f}'.format(max(theta)), file=out_f)
        print('Nonzero phi at optimum (#): {:d}'.format(np.count_nonzero(phi)), file=out_f)
        print('Max phi at optimum: {:.4f}'.format(max(phi)), file=out_f)
        y_names = ['y_' + str(i) for i in range(graph.num_nodes)]
        values = best_solution
        seeds = [i for i in range(graph.num_nodes) if values[i] > 0.5]

    if debugging:
        print('Solution:', file=out_f)
        print(values, file=out_f)
        print('Chosen seeds:', file=out_f)
        print(seeds, file=out_f)

    results = [time_elapsed, nodes_visited, mip_gap*100, best_bound,
               best_objective, num_covers, len(lazy_constraints),
               np.count_nonzero(theta), max(theta),
               np.count_nonzero(phi), max(phi), iterations]


    prob.end()

    return results
# -- end function

def add_columns(prob, graph, cover_pointer, thresholds, generated_covers,
                generated_indices, current_obj, debugging):

    # We should add them in the same order they were added during column generation
    d = dict()
    for i in range(graph.num_nodes):
        for k, cover in enumerate(generated_covers[i]):
            d[generated_indices[i][k]] = (i, cover)
    
    od = collections.OrderedDict(sorted(d.items()))

    # Handy pointer
    phi_pointer = dict()
    ind = 0
    for i in range(graph.num_nodes):
        for j in graph.instar[i]:
            phi_pointer[i, j] = ind
            ind += 1

    for k, v in od.items():
        i, cover = v
        node, index = cover_pointer[k]
        # Compute coefficient objective function
        coefficient = len(cover) - 1
        (_, theta, phi) = current_obj[1]
        # Get the sum of the optimal phi associated with this cover
        phi_cover_sum = sum(phi[phi_pointer[node, l]] for l in cover)
        curr_thresh = graph.node_threshold[node] + theta[node] + phi_cover_sum
        if (curr_thresh > thresholds[node][index][1]):
            coefficient += 1
        cover_dual_ind = [h for h in cover] + [i]
        cover_dual_val = [1 for _ in cover] + [-1]
        if (debugging):
            print('Adding cover {:s} for node {:d}: '.format('pi_' + str(k), i), cover)
        prob.variables.add(
            lb=[-cplex.infinity],
            ub=[0],
            obj=[coefficient],
            types=[prob.variables.type.continuous],
            names=['pi_' + str(k)],
            columns=[cplex.SparsePair(ind=cover_dual_ind, val=cover_dual_val)])

