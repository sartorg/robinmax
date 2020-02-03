""" Column generation

"""

import numpy as np
import cplex
import time
import sys

import robinmax_pricing as pr

def column_generation(graph, incumbent_info, pruned_info, covers, thresholds,
                      thresh_budget, max_thresh_dev, weight_budget,
                      max_weight_dev, max_time, epsilon, debugging, out_f):

    start_time = time.time()

    prob = cplex.Cplex()
    prob.parameters.timelimit.set(max_time)

    prob.parameters.threads.set(1)
    prob.parameters.emphasis.numerical.set(1)

    prob.set_log_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    # We need to give each cover a unique index
    cover_index = [list() for _ in range(graph.num_nodes)]
    # For each cover, we store its indices (i, j) to find the
    # corresponding set in covers
    cover_pointer = list()
    cover_size = list()
    num_covers = 0
    for i in range(graph.num_nodes):
        # Look at how many covers are associated with this node, and
        # assign to them the indices from num_covers to num_covers
        # plus the number of covers at this node
        cover_index[i] = [num_covers + j for j in range(len(covers[i]))]
        for j in range(len(cover_index[i])):
            cover_pointer.append([i, j])
            cover_size.append(len(covers[i][j]))
        num_covers += len(cover_index[i])

    if debugging:
        for i in range(graph.num_nodes):
            print('Node', i, file=out_f)
            print('Covers:', covers[i], file=out_f)
            print('Cover indices:', cover_index[i], file=out_f)

    if (debugging):
        print('Found {:d} covers.\n'.format(num_covers), file=out_f)

    # Add pi variable
    pi_start = 0
    prob.variables.add(
        lb=[-cplex.infinity] * num_covers,
        ub=[0] * num_covers,
        types=[prob.variables.type.continuous] * num_covers,
        names=['pi_' + str(i) for i in range(num_covers)])
    # Add mu variables
    mu_start = pi_start + num_covers
    prob.variables.add(
        lb=[0] * graph.num_nodes,
        ub=[cplex.infinity] * graph.num_nodes,
        types=[prob.variables.type.continuous] * graph.num_nodes,
        names=['mu_' + str(i) for i in range(graph.num_nodes)])
    # Add variable that represents the obj
    z_start = mu_start + graph.num_nodes
    prob.variables.add(obj=[1], lb=[0], ub = [graph.num_nodes],
                        types=[prob.variables.type.continuous], names=['z'])
            
    # Add lazy constraints
    phi_pointer = dict()
    ind = 0
    for i in range(graph.num_nodes):
        for j in graph.instar[i]:
            phi_pointer[i, j] = ind
            ind += 1

    for lazy_i, (incumbent, theta, phi) in enumerate(incumbent_info):
        # Initialize the coefficients of the cover vars to |S| - 1
        coefficients = [val - 1 for val in cover_size]
        for i in range(num_covers):
                node, index = cover_pointer[i]
                # Get the sum of the optimal phi associated with this cover
                phi_cover_sum = sum(phi[phi_pointer[node, j]] 
                            for j in covers[node][index])
                curr_thresh = graph.node_threshold[node] + theta[node] + phi_cover_sum
                if (#curr_thresh <= self.thresholds[node][index][0] or
                    curr_thresh > thresholds[node][index][1]):
                    coefficients[i] = coefficients[i] + 1

        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=[i for i in range(graph.num_nodes +
                                    num_covers + 1)],
                val=coefficients + [1] * graph.num_nodes + [-1])],
                senses=['G'], rhs=[0],
                names=['lazy_constraint' + str(lazy_i)])

    # Adding cover dual.
    for i in range(graph.num_nodes):
        # The pi variables with coefficient +1 are those for covers
        # that node i can activate.
        pos_ind = sorted(pi_start + cover_index[j][k]
                         for j in graph.outstar[i]
                         for k in range(len(covers[j]))
                         if i in covers[j][k])
        # The pi variables with coefficient -1 are those for covers
        # that activate node i.
        neg_ind = [pi_start + cover_index[i][k]
                   for k in range(len(covers[i]))]

        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=pos_ind + neg_ind +
                                         [mu_start + i],
                                         val=[1] * len(pos_ind) +
                                         [-1] * len(neg_ind) + [1])],
            senses=['L'], rhs=[1], names=['cover_dual_' + str(i)])

    # Objective funciton
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[i for i in range(z_start + 1)],
                                   val=[cover_size[i] - 1 for i in range(num_covers)] 
                                        + [1] * graph.num_nodes + [-1])],
        senses=['G'], rhs=[0],
        names=['objective_function'])
    
    # This is a LP
    prob.set_problem_type(prob.problem_type.LP)

    # This is a maximization problem
    prob.objective.set_sense(prob.objective.sense.maximize)

    best_obj = 0

    num_generated_covers = 0

    for lazy_i, (incumbent, theta, phi) in enumerate(incumbent_info):

        if (time.time() - start_time > max_time):
            break

        # Use incumbent to fix mu
        for j in range(graph.num_nodes):
            if (round(incumbent[j]) == 0):
                prob.variables.set_upper_bounds(mu_start + j, 0)
            else:
                prob.variables.set_upper_bounds(mu_start + j, cplex.infinity)

        any_generated_cover = True
        iterations = 0
        while(any_generated_cover and time.time() - start_time <= max_time):
            iterations += 1
            if (debugging):
                print('LP iteration: ' + str(iterations))

            # Apparently we have to keep remembering CPLEX this is a LP.
            prob.set_problem_type(prob.problem_type.LP)

            time_left = max(0, (max_time - (time.time() - start_time)))
            prob.parameters.timelimit.set(time_left)

            # Solve
            prob.parameters.threads.set(1)
            prob.solve()

            this_obj = prob.solution.get_objective_value()
            if (debugging):
                print('LP objective value: ' + str(this_obj))
            if (round(this_obj) > round(best_obj)):
                best_obj = this_obj
                if (debugging):
                    print('Found new LP bound with value: ', this_obj)

            #prob.write('lp_column' + str(pruned_i) + '.lp')
            # Get dual variables
            dual_variables_val = prob.solution.get_dual_values()
            lambda_val = dual_variables_val[:len(incumbent_info)]
            x_val = dual_variables_val[len(incumbent_info):]

            time_left = max(0, (max_time - (time.time() - start_time)))
            # Solve pricing problem
            generated_covers = pr.pricing_problem(graph, incumbent_info, phi_pointer, 
                                                covers, lambda_val, x_val, epsilon,
                                                thresh_budget, max_thresh_dev, max_weight_dev,
                                                time_left, debugging)
            any_generated_cover = False
            new_generated_covers = 0
            # Add columns
            for i in range(graph.num_nodes):
                if (len(generated_covers[i]) > 0):
                    # We generated at least on column
                    any_generated_cover = True
                    # Adding the column to the LP
                    for k, cover in enumerate(generated_covers[i]):
                        # Updating the covers set
                        covers[i].append(cover)
                        # Updating threshold
                        cover_weights = [graph.arc_weight_in[i][h] for h in range(len(graph.arc_weight_in[i]))
                                         if graph.instar[i][h] in cover]
                        value = sum(cover_weights)
                        minimality = value - min(cover_weights)
                        thresholds[i].append((minimality, value))
                        # Check if it is valid and update coefficient
                        coefficients = [len(cover) - 1 for _ in range(len(incumbent_info))]
                        for h, (incumbent, theta, phi) in enumerate(incumbent_info):
                            # Get the sum of the optimal phi associated with this cover
                            phi_cover_sum = sum(phi[phi_pointer[i, l]] for l in cover)
                            curr_thresh = graph.node_threshold[i] + theta[i] + phi_cover_sum
                            if (curr_thresh > value):
                                coefficients[h] = coefficients[h] + 1
                        cover_dual_ind = [h + len(incumbent_info) for h in cover] + [i + len(incumbent_info)]
                        cover_dual_val = [1 for _ in cover] + [-1]
                        obj_ind = [len(incumbent_info) + graph.num_nodes]
                        obj_val = [len(cover) - 1]
                        prob.variables.add(
                            lb=[-cplex.infinity],
                            ub=[0],
                            types=[prob.variables.type.continuous],
                            names=['pi_' + str(num_covers + k)],
                            columns=[cplex.SparsePair(ind=[h for h in range(len(incumbent_info))] + 
                                                           cover_dual_ind + obj_ind, 
                                                      val=coefficients + cover_dual_val + obj_val)])
                    num_covers += len(generated_covers[i])
                    new_generated_covers += len(generated_covers[i])
            num_generated_covers += new_generated_covers
            if (debugging):
                print('Number of generated covers: ' + str(new_generated_covers))

    if (num_generated_covers == 0):
        # If we already have new covers we do not want to loop over
        # pruned points
        num_covers_before_pruned = num_generated_covers
        for pruned_i, pruned_point in enumerate(reversed(pruned_info)):

            if (time.time() - start_time > max_time):
                break
            
            # Use incumbent to fix mu
            for j in range(graph.num_nodes):
                if (abs(pruned_point[j]) <= 1.0e-8):
                    prob.variables.set_upper_bounds(mu_start + j, 0)
                else:
                    prob.variables.set_upper_bounds(mu_start + j, cplex.infinity)

            any_generated_cover = True
            iterations = 0
            while(any_generated_cover and time.time() - start_time <= max_time):
                iterations += 1
                if (debugging):
                    print('LP iteration: ' + str(iterations))

                # Apparently we have to keep remembering CPLEX this is a LP.
                prob.set_problem_type(prob.problem_type.LP)
                
                time_left = max(0, (max_time - (time.time() - start_time)))
                prob.parameters.timelimit.set(time_left)

                # Solve
                prob.parameters.threads.set(1)
                prob.solve()

                this_obj = prob.solution.get_objective_value()
                if (debugging):
                    print('LP objective value: ' + str(this_obj))
                if (round(this_obj) > round(best_obj)):
                    best_obj = this_obj
                    print('Found new LP bound with value: ', this_obj)

                #prob.write('lp_column' + str(lazy_i) + '.lp')
                # Get dual variables
                dual_variables_val = prob.solution.get_dual_values()
                lambda_val = dual_variables_val[:len(incumbent_info)]
                x_val = dual_variables_val[len(incumbent_info):]
                # Solve pricing problem
                time_left = max(0, (max_time - (time.time() - start_time)))
                generated_covers = pr.pricing_problem(
                    graph, incumbent_info, phi_pointer, 
                    covers, lambda_val, x_val, epsilon,
                    thresh_budget, max_thresh_dev, max_weight_dev,
                    time_left, debugging)
                any_generated_cover = False
                new_generated_covers = 0
                # Add columns
                for i in range(graph.num_nodes):
                    if (len(generated_covers[i]) > 0):
                        # We generated at least on column
                        any_generated_cover = True
                        # Adding the column to the LP
                        for k, cover in enumerate(generated_covers[i]):
                            # Updating the covers set
                            covers[i].append(cover)
                            # Updating threshold
                            cover_weights = [graph.arc_weight_in[i][h] for h in range(len(graph.arc_weight_in[i]))
                                             if graph.instar[i][h] in cover]
                            value = sum(cover_weights)
                            minimality = value - min(cover_weights)
                            thresholds[i].append((minimality, value))
                            # Check if it is valid and update coefficient
                            coefficients = [len(cover) - 1 for _ in range(len(incumbent_info))]
                            for h, (incumbent, theta, phi) in enumerate(incumbent_info):
                                # Get the sum of the optimal phi associated with this cover
                                phi_cover_sum = sum(phi[phi_pointer[i, l]] for l in cover)
                                curr_thresh = graph.node_threshold[i] + theta[i] + phi_cover_sum
                                if (curr_thresh > value):
                                    coefficients[h] = coefficients[h] + 1
                            cover_dual_ind = [h + len(incumbent_info) for h in cover] + [i + len(incumbent_info)]
                            cover_dual_val = [1 for _ in cover] + [-1]
                            obj_ind = [len(incumbent_info) + graph.num_nodes]
                            obj_val = [len(cover) - 1]
                            prob.variables.add(
                                lb=[-cplex.infinity],
                                ub=[0],
                                types=[prob.variables.type.continuous],
                                names=['pi_' + str(num_covers + k)],
                                columns=[cplex.SparsePair(ind=[h for h in range(len(incumbent_info))] + 
                                                          cover_dual_ind + obj_ind, 
                                                          val=coefficients + cover_dual_val + obj_val)])
                        num_covers += len(generated_covers[i])
                        new_generated_covers += len(generated_covers[i])
                num_generated_covers += new_generated_covers
                if (debugging):
                    print('Number of generated covers: ' + str(new_generated_covers))
            if (num_generated_covers > num_covers_before_pruned):
                break
        prob.end()

    return num_generated_covers, best_obj
            
# -- end function


def column_generation2(graph, incumbent_info, covers, thresholds,
                       cover_index, cover_pointer, cover_size, num_covers,
                       thresh_budget, max_thresh_dev, weight_budget,
                       max_weight_dev, max_time, epsilon, current_obj,
                       current_best_obj, pricing_problems, max_pricing_iters,
                       max_iters_per_round, can_stop_early=False, 
                       max_columns_per_round=10000,
                       debugging=False, out_f=sys.__stdout__):

    start_time = time.time()

    # Handy pointers
    phi_pointer = dict()
    ind = 0
    for i in range(graph.num_nodes):
        for j in graph.instar[i]:
            phi_pointer[i, j] = ind
            ind += 1
    psi_pointer = dict()
    for i in range(graph.num_nodes):
        for j, ind in enumerate(graph.instar[i]):
            psi_pointer[i, ind] = j

    prob = cplex.Cplex()
    prob.parameters.timelimit.set(max_time)

    prob.parameters.threads.set(1)
    prob.parameters.emphasis.numerical.set(1)

    prob.set_log_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)

    # if debugging:
    #     for i in range(graph.num_nodes):
    #         print('Node', i, file=out_f)
    #         print('Covers:', covers[i], file=out_f)
    #         print('Cover indices:', cover_index[i], file=out_f)

    if (debugging):
        print('Column generation starting with {:d} covers.\n'.format(num_covers), file=out_f)

    # Add pi variable
    pi_start = 0
    prob.variables.add(
        lb=[-cplex.infinity] * num_covers,
        ub=[0] * num_covers,
        types=[prob.variables.type.continuous] * num_covers,
        names=['pi_' + str(i) for i in range(num_covers)])
    # Add mu variables
    mu_start = pi_start + num_covers
    prob.variables.add(
        lb=[0] * graph.num_nodes,
        ub=[cplex.infinity] * graph.num_nodes,
        types=[prob.variables.type.continuous] * graph.num_nodes,
        names=['mu_' + str(i) for i in range(graph.num_nodes)])
            
    # Handy pointer
    phi_pointer = dict()
    ind = 0
    for i in range(graph.num_nodes):
        for j in graph.instar[i]:
            phi_pointer[i, j] = ind
            ind += 1

    # Adding cover dual.
    for i in range(graph.num_nodes):
        # The pi variables with coefficient +1 are those for covers
        # that node i can activate.
        pos_ind = sorted(pi_start + cover_index[j][k]
                         for j in graph.outstar[i]
                         for k in range(len(covers[j]))
                         if i in covers[j][k])
        # The pi variables with coefficient -1 are those for covers
        # that activate node i.
        neg_ind = [pi_start + cover_index[i][k]
                   for k in range(len(covers[i]))]

        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=pos_ind + neg_ind +
                                         [mu_start + i],
                                         val=[1] * len(pos_ind) +
                                         [-1] * len(neg_ind) + [1])],
            senses=['L'], rhs=[1], names=['cover_dual_' + str(i)])

    # Objective funciton
    _, theta, phi = current_obj[1]
    # Initialize the coefficients of the cover vars to |S| - 1
    coefficients = [val - 1 for val in cover_size]
    for i in range(num_covers):
        node, index = cover_pointer[i]
        # Get the sum of the optimal phi associated with this cover
        phi_cover_sum = sum(phi[phi_pointer[node, j]] 
                    for j in covers[node][index])
        curr_thresh = graph.node_threshold[node] + theta[node] + phi_cover_sum
        if (curr_thresh > thresholds[node][index][1]):
            coefficients[i] = coefficients[i] + 1

    ind = [i for i in range(num_covers + graph.num_nodes)]
    val = coefficients + [1] * graph.num_nodes
    prob.objective.set_linear([(ind[i], val[i]) for i in range(len(ind))])
    
    # This is a LP
    prob.set_problem_type(prob.problem_type.LP)

    # This is a maximization problem
    prob.objective.set_sense(prob.objective.sense.maximize)

    best_obj = 0

    num_generated_covers = 0
    all_generated_covers = [list() for _ in range(graph.num_nodes)]
    all_generated_indices = [list() for _ in range(graph.num_nodes)]

    if (debugging):
        print("Generating columns for {:d} incumbent(s).".format(len(incumbent_info)))

    for lazy_i, (incumbent, _, _) in enumerate(incumbent_info):
        if (debugging):
            print("Generating columns for incumbent {:d}: ".format(lazy_i),
                [i for i in range(graph.num_nodes) if incumbent[i] >= 0.5])

        if (time.time() - start_time > max_time):
            break

        # Use incumbent to fix mu
        for j in range(graph.num_nodes):
            if (round(incumbent[j]) == 0):
                prob.variables.set_upper_bounds(mu_start + j, 0)
            else:
                prob.variables.set_upper_bounds(mu_start + j, cplex.infinity)

        any_generated_cover = True
        iterations = 0
        while(any_generated_cover and time.time() - start_time <= max_time and
              best_obj <= current_best_obj - 1.0e-8 and
              num_generated_covers < max_columns_per_round and
              (iterations < max_iters_per_round or not can_stop_early)):
            iterations += 1
            if (debugging):
                print('LP iteration: ' + str(iterations))

            # Apparently we have to keep remembering CPLEX this is a LP.
            prob.set_problem_type(prob.problem_type.LP)

            time_left = max(0, (max_time - (time.time() - start_time)))
            prob.parameters.timelimit.set(time_left)

            # Solve
            prob.parameters.threads.set(1)
            prob.solve()
            #prob.write("last_lp_CG.lp")

            this_obj = prob.solution.get_objective_value()
            if (debugging ):
                print('Column generation LP objective value: ' + str(this_obj))
            if (round(this_obj) > round(best_obj)):
                best_obj = this_obj
                if (debugging):
                    print('Found new LP bound with value: ', this_obj)

            #if (num_generated_covers > 0 and
            #    best_obj >= current_best_obj - 1.0e-8):
            #    break

            #prob.write('lp_column' + str(pruned_i) + '.lp')
            # Get dual variables
            x_val = prob.solution.get_dual_values()

            time_left = max(0, (max_time - (time.time() - start_time)))
            # Solve pricing problem            
            generated_covers = pr.pricing_problem3(
                graph, current_obj[1], phi_pointer, 
                covers, x_val, epsilon,
                thresh_budget, max_thresh_dev, max_weight_dev,
                time_left, psi_pointer, pricing_problems,
                max_pricing_iters, debugging)
            any_generated_cover = False
            new_generated_covers = 0
            # Add columns
            for i in range(graph.num_nodes):
                if (len(generated_covers[i]) > 0):
                    # We generated at least on column
                    any_generated_cover = True
                    # Adding the column to the LP
                    for k, cover in enumerate(generated_covers[i]):
                        # Updating the covers set
                        covers[i].append(cover)
                        # Updating threshold
                        cover_weights = [
                            graph.arc_weight_in[i][h]
                            for h in range(len(graph.arc_weight_in[i]))
                            if graph.instar[i][h] in cover]
                        value = sum(cover_weights)
                        minimality = value - min(cover_weights)
                        thresholds[i].append((minimality, value))
                        # Updating cover index
                        cover_index[i].append(num_covers + k)
                        # Updating cover pointer
                        cover_pointer.append([i, len(cover_index[i]) - 1])
                        # Updating conver size
                        cover_size.append(len(cover))
                        # Update the list and indices of all generated covers
                        all_generated_covers[i].append(cover)
                        all_generated_indices[i].append(cover_index[i][-1])
                        # Check if it is valid and update coefficient
                        coefficient = len(cover) - 1
                        (_, theta, phi) = current_obj[1]
                        # Get the sum of the optimal phi associated with this cover
                        phi_cover_sum = sum(phi[phi_pointer[i, l]] for l in cover)
                        curr_thresh = graph.node_threshold[i] + theta[i] + phi_cover_sum
                        if (curr_thresh > value):
                            coefficient += 1
                        cover_dual_ind = [h for h in cover] + [i]
                        cover_dual_val = [1 for _ in cover] + [-1]
                        obj_ind = [graph.num_nodes]
                        obj_val = [coefficient]
                        prob.variables.add(
                            lb=[-cplex.infinity],
                            ub=[0],
                            obj=[coefficient],
                            types=[prob.variables.type.continuous],
                            names=['pi_' + str(num_covers + k)],
                            columns=[cplex.SparsePair(ind=cover_dual_ind, 
                                                      val=cover_dual_val)])
                    num_covers += len(generated_covers[i])
                    new_generated_covers += len(generated_covers[i])
            num_generated_covers += new_generated_covers
            if (debugging):
                print('Number of generated covers: ' +
                      str(new_generated_covers))
            #if (iterations > 0 and iterations % 100 == 0):
            #    print('Iter {:d} CG LP obj {:f} covers {:d}'.format(
            #        iterations, this_obj, num_generated_covers))
    prob.end()

    return num_generated_covers, best_obj, all_generated_covers, all_generated_indices
            
# -- end function
