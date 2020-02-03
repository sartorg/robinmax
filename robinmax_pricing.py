""" Generating pricing problem


"""

import numpy as np
import cplex
import time

def pricing_problem(graph, incumbent_info, phi_pointer, covers,
                    lambda_val, x_val, epsilon, thresh_budget,
                    max_thresh_dev, max_weight_dev, max_time,
                    debugging):

    start_time = time.time()

    # Initialize the generated cover list
    generated_covers = []

    # Solve a pricing problem for each node, i.e., for each node,
    # try to find a nondegenerate cover of that node with positive
    # reduced cost.
    for j in range(graph.num_nodes):

        prob = cplex.Cplex()

        prob.parameters.threads.set(1)
        prob.parameters.emphasis.numerical.set(1)

        prob.set_log_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        shift_obj = x_val[j] + sum(lambda_val)

        # Add psi variables
        psi_pointer = dict()
        for i, ind in enumerate(graph.instar[j]):
            psi_pointer[ind] = i
        psi_start = 0
        prob.variables.add(
            obj=[x_val[i] + sum(lambda_val) for i in range(len(graph.instar[j]))],
            types=[prob.variables.type.binary for i in range(len(graph.instar[j]))],
            names=['psi_' + str(i) for i in range(len(graph.instar[j]))])

        # Add beta variables
        beta_start = psi_start + len(graph.instar[j])
        prob.variables.add(
            obj=[lambda_val[i] for i in range(len(incumbent_info))],
            types=[prob.variables.type.binary for i in range(len(incumbent_info))],
            names=['beta_' + str(i) for i in range(len(incumbent_info))])

        # Add alpha variable
        alpha_start = beta_start + len(incumbent_info)
        prob.variables.add(
            lb=[0],
            ub=[cplex.infinity],
            types=[prob.variables.type.continuous],
            names=['alpha'])

        # Add weights constraint
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))],
                                       val=[graph.arc_weight_in[j][i] 
                                        for i in range(len(graph.arc_weight_in[j]))])],
            senses=['G'], rhs=[graph.node_threshold[j]],
            names=['weight cover'])

        # Add cover validity constraints
        for lazy_i, (incumbent, theta, phi) in enumerate(incumbent_info):
            psi_ind = [i for i in range(len(graph.instar[j]))]

            psi_val = [graph.arc_weight_in[j][i] - phi[phi_pointer[j, graph.instar[j][i]]]
                                                for i in range(len(graph.arc_weight_in[j]))]
            beta_ind = [beta_start + lazy_i]
            beta_val = [graph.node_threshold[j] + theta[j]]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=psi_ind + beta_ind,
                                           val=psi_val + beta_val)],
                senses=['G'], rhs=[graph.node_threshold[j] + theta[j]],
                names=['validity_cover' + str(lazy_i)])

        # Add linearization minimum cover weigth
        max_weight_in = max(graph.arc_weight_in[j])
        for i in range(len(graph.instar[j])):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[psi_start + i] + [alpha_start],
                                           val=[max_weight_in - (1 - max_weight_dev) * graph.arc_weight_in[j][i]] + [1])],
                senses=['L'], rhs=[max_weight_in],
                names=['minimum_' + str(i)])

        # Add cover minimality
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))] + [alpha_start],
                                       val=[graph.arc_weight_in[j][i] * (1 - max_weight_dev)
                                        for i in range(len(graph.arc_weight_in[j]))] + [-1])],
            senses=['L'], rhs=[graph.node_threshold[j] + min(thresh_budget, max_thresh_dev * graph.node_threshold[j])],
            names=['minimality'])
        
        # Initialize dgenerate covers list
        degenerate_covers = covers[j]
        new_covers = []
        any_degenerate_cover = True
        #while(any_degenerate_cover):
        for d in range(1):
            # Add nogood cuts to invalidate degenerate covers
            for i, degenerate_cover in enumerate(degenerate_covers):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[psi_pointer[k] for k in degenerate_cover],
                                            val=[1] * len(degenerate_cover))],
                    senses=['L'], rhs=[len(degenerate_cover) - 1],
                    names=['degenerate_cover' + str(i)])

            # This is a maximization problem
            prob.objective.set_sense(prob.objective.sense.maximize)

            time_left = max(0, (max_time - (time.time() - start_time)))
            prob.parameters.timelimit.set(time_left)
            prob.parameters.threads.set(1) 
            prob.solve()

            #prob.write('pricing.lp')
            #print(degenerate_covers)

            any_degenerate_cover = False
            for i in range(prob.solution.pool.get_num()):
                # Get objective value
                obj = prob.solution.pool.get_objective_value(i)
                # Check if the reduced cost is strictly positive, i.e.,
                # if the objective of the pricing problem (minus a shift)
                # is strictly positive
                if (obj >= shift_obj + 1e-8):
                    #print('Obj: ', obj)
                    #print('Shift', shift_obj)
                    #prob.write('pricing.lp')
                    #exit()
                    solution = prob.solution.pool.get_values(i)
                    new_cover = []
                    for k in range(len(graph.instar[j])):
                        if (round(solution[k]) == 1):
                            new_cover.append(graph.instar[j][k])
                    if (len(new_cover) == 0):
                        raise RuntimeError('Pricing problem found an empty cover.')
                    # Check if we found a degenerate cover
                    if (new_cover in covers[j] or new_cover in new_covers):
                        any_degenerate_cover = True
                        if (debugging):
                            print('Found degenerate cover for node {:d}: '.format(j), new_cover)
                        #degenerate_covers.append(new_cover)
                    else:
                        if (debugging):
                            print('Found new cover for node {:d}: '.format(j), new_cover)
                        new_covers.append(new_cover)

        generated_covers.append(new_covers)

        prob.end()

    return generated_covers


def pricing_problem2(graph, incumbent_info, phi_pointer, covers,
                     x_val, epsilon, thresh_budget,
                     max_thresh_dev, max_weight_dev, max_time,
                     debugging):

    start_time = time.time()

    # Initialize the generated cover list
    generated_covers = [[] for j in range(graph.num_nodes)]
    num_generated_covers = 0

    # Randomly choose order of pricing
    node_order = np.random.permutation(graph.num_nodes)

    # Solve a pricing problem for each node, i.e., for each node,
    # try to find a nondegenerate cover of that node with positive
    # reduced cost.
    iteration = 0
    while ((iteration < graph.num_nodes*0.1 or num_generated_covers == 0)
           and iteration < graph.num_nodes):
        j = node_order[iteration]
        iteration += 1

        prob = cplex.Cplex()

        prob.parameters.threads.set(1)
        prob.parameters.emphasis.numerical.set(1)

        prob.set_log_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        shift_obj = x_val[j] + 1

        # Add psi variables
        psi_pointer = dict()
        for i, ind in enumerate(graph.instar[j]):
            psi_pointer[ind] = i
        psi_start = 0
        prob.variables.add(
            obj=[x_val[i] + 1 for i in range(len(graph.instar[j]))],
            types=[prob.variables.type.binary for i in range(len(graph.instar[j]))],
            names=['psi_' + str(i) for i in range(len(graph.instar[j]))])

        # Add beta variable
        beta_start = psi_start + len(graph.instar[j])
        prob.variables.add(
            obj=[1],
            types=[prob.variables.type.binary],
            names=['beta'])

        # Add alpha variable
        alpha_start = beta_start + 1
        prob.variables.add(
            lb=[0],
            ub=[cplex.infinity],
            types=[prob.variables.type.continuous],
            names=['alpha'])

        # Add weights constraint
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))],
                                       val=[graph.arc_weight_in[j][i] 
                                        for i in range(len(graph.arc_weight_in[j]))])],
            senses=['G'], rhs=[graph.node_threshold[j]],
            names=['weight cover'])

        # Add cover validity constraints
        (_, theta, phi) = incumbent_info
        psi_ind = [i for i in range(len(graph.instar[j]))]
        psi_val = [graph.arc_weight_in[j][i] - phi[phi_pointer[j, graph.instar[j][i]]]
                                            for i in range(len(graph.arc_weight_in[j]))]
        beta_ind = [beta_start]
        beta_val = [graph.node_threshold[j] + theta[j]]
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=psi_ind + beta_ind,
                                        val=psi_val + beta_val)],
            senses=['G'], rhs=[graph.node_threshold[j] + theta[j]],
            names=['validity_cover'])

        # Add linearization minimum cover weigth
        max_weight_in = max(graph.arc_weight_in[j])
        for i in range(len(graph.instar[j])):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[psi_start + i] + [alpha_start],
                                           val=[max_weight_in - graph.arc_weight_in[j][i] - phi[phi_pointer[j, graph.instar[j][i]]]] + [1])],
                senses=['L'], rhs=[max_weight_in],
                names=['minimum_' + str(i)])

        # Add cover minimality
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))] + [alpha_start],
                                       val=[graph.arc_weight_in[j][i] - phi[phi_pointer[j, graph.instar[j][i]]]
                                        for i in range(len(graph.arc_weight_in[j]))] + [-1])],
            senses=['L'], rhs=[graph.node_threshold[j] + theta[j]],
            names=['minimality'])
        
        # Initialize dgenerate covers list
        degenerate_covers = covers[j]
        new_covers = []
        any_degenerate_cover = True
        # Add nogood cuts to invalidate degenerate covers
        for i, degenerate_cover in enumerate(degenerate_covers):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[k for k in range(len(graph.instar[j]))],
                                           val=[(-1 if graph.instar[j][k] in degenerate_cover else 1) for k in range(len(graph.instar[j]))])],
                senses=['G'], rhs=[1 - len(degenerate_cover)],
                names=['degenerate_cover' + str(i)])

        # This is a maximization problem
        prob.objective.set_sense(prob.objective.sense.maximize)
        
        time_left = max(0, (max_time - (time.time() - start_time)))
        prob.parameters.timelimit.set(time_left)
        prob.parameters.threads.set(1) 
        prob.solve()
        #prob.write('last_pricing.lp')
        #print(degenerate_covers)

        any_degenerate_cover = False
        for i in range(prob.solution.pool.get_num()):
            # Get objective value
            obj = prob.solution.pool.get_objective_value(i)
            # Check if the reduced cost is strictly positive, i.e.,
            # if the objective of the pricing problem (minus a shift)
            # is strictly positive
            if (obj >= shift_obj + 1e-8):
                solution = prob.solution.pool.get_values(i)
                new_cover = []
                for k in range(len(graph.instar[j])):
                    if (round(solution[k]) == 1):
                        new_cover.append(graph.instar[j][k])
                if (len(new_cover) == 0):
                    raise RuntimeError('Pricing problem found an empty cover.')
                # Check if we found a degenerate cover
                if (new_cover in covers[j] or new_cover in new_covers):
                    any_degenerate_cover = True
                    if (debugging):
                        print('Found degenerate cover for node {:d}: '.format(j), new_cover)
                    #degenerate_covers.append(new_cover)
                else:
                    if (debugging):
                        print('Found new cover for node {:d} reduced cost {:f}: '.format(j, obj-shift_obj), new_cover)
                    new_covers.append(new_cover)

        generated_covers[j].extend(new_covers)
        num_generated_covers += len(new_covers)

        prob.end()

    return generated_covers
# -- end function

def pricing_problem3(graph, incumbent_info, phi_pointer, covers,
                     x_val, epsilon, thresh_budget,
                     max_thresh_dev, max_weight_dev, max_time,
                     psi_pointer, pricing_problems,
                     max_pricing_iters, debugging):

    start_time = time.time()

    # Initialize the generated cover list
    generated_covers = [[] for j in range(graph.num_nodes)]
    num_generated_covers = 0

    # Randomly choose order of pricing
    node_order = np.random.permutation(graph.num_nodes)

    # Solve a pricing problem for each node, i.e., for each node,
    # try to find a nondegenerate cover of that node with positive
    # reduced cost.
    iteration = 0
    while ((iteration < max_pricing_iters or num_generated_covers == 0)
           and iteration < graph.num_nodes):
        j = node_order[iteration]
        iteration += 1

        # Get the CPLEX problem
        prob = pricing_problems[j]

        # Now we need to update the problem based on the current incumbent

        # First, update the objective
        prob.objective.set_linear([(i, x_val[i] + 1)
                                   for i in range(len(graph.instar[j]))])

        # Second, update the cover validity constraint
        (_, theta, phi) = incumbent_info
        psi_ind = [i for i in range(len(graph.instar[j]))]
        psi_val = [graph.arc_weight_in[j][i] - phi[phi_pointer[j, graph.instar[j][i]]]
                                            for i in range(len(graph.arc_weight_in[j]))]
        beta_ind = [len(graph.instar[j])]
        beta_val = [graph.node_threshold[j] + theta[j]]
        prob.linear_constraints.set_linear_components('validity_cover',
            cplex.SparsePair(ind=psi_ind + beta_ind,
                             val=psi_val + beta_val))
        prob.linear_constraints.set_rhs('validity_cover', graph.node_threshold[j] + theta[j])

        # Update linearization minimum cover weigth
        max_weight_in = max(graph.arc_weight_in[j])
        for i in range(len(graph.instar[j])):
            prob.linear_constraints.set_linear_components(
                'minimum_'  + str(i),
                cplex.SparsePair(
                    ind=[i] +
                    [len(graph.instar[j]) + 1],
                    val=[max_weight_in - graph.arc_weight_in[j][i] -
                         phi[phi_pointer[j, graph.instar[j][i]]]] + [1]))

        # Update cover minimality
        prob.linear_constraints.set_linear_components(
            'minimality',
            cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))] +
                             [len(graph.instar[j]) + 1],
                             val=[graph.arc_weight_in[j][i] -
                                  phi[phi_pointer[j, graph.instar[j][i]]]
                                  for i in range(len(graph.arc_weight_in[j]))]
                             + [-1]))
        prob.linear_constraints.set_rhs('minimality', graph.node_threshold[j] + theta[j])
        
        # This is a maximization problem
        prob.objective.set_sense(prob.objective.sense.maximize)

        time_left = max(0, (max_time - (time.time() - start_time)))
        prob.parameters.timelimit.set(time_left)
        prob.parameters.threads.set(1) 
        prob.solve()
        #prob.write('last_pricing.lp')

        # Compute the shift of the objective function
        shift_obj = x_val[j] + 1

        new_covers = []
        for i in range(prob.solution.pool.get_num()):
            # Get objective value
            obj = prob.solution.pool.get_objective_value(i)
            # Check if the reduced cost is strictly positive, i.e.,
            # if the objective of the pricing problem (minus a shift)
            # is strictly positive
            if (obj >= shift_obj + 1e-8):
                solution = prob.solution.pool.get_values(i)
                new_cover = []
                for k in range(len(graph.instar[j])):
                    if (round(solution[k]) == 1):
                        new_cover.append(graph.instar[j][k])
                if (len(new_cover) == 0):
                    raise RuntimeError('Pricing problem found an empty cover.')
                # Check if we found a degenerate cover
                if (new_cover in covers[j] or new_cover in new_covers):
                    if (debugging):
                        print('Found degenerate cover for node {:d}: '.format(j), new_cover)
                else:
                    if (debugging):
                        print('Found new cover for node {:d}: '.format(j), new_cover)
                    new_covers.append(new_cover)

        # Update the problem with generated covers for the next iteration
        for i, nogood_cover in enumerate(new_covers):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[k for k in range(len(graph.instar[j]))],
                                           val=[(-1 if graph.instar[j][k] in nogood_cover else 1) for k in range(len(graph.instar[j]))])],
                senses=['G'], rhs=[1 - len(nogood_cover)],
                names=['nogood_cover' + str(i)])

        generated_covers[j].extend(new_covers)
        num_generated_covers += len(new_covers)

    return generated_covers


def create_pricing_problems(graph, thresh_budget, max_thresh_dev,
                            max_weight_dev, debugging):

    pricing_problems = list()

    for j in range(graph.num_nodes):

        prob = cplex.Cplex()

        prob.parameters.threads.set(1)
        prob.parameters.emphasis.numerical.set(1)

        prob.set_log_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        # Add psi variables
        psi_start = 0
        prob.variables.add(
            types=[prob.variables.type.binary for i in range(len(graph.instar[j]))],
            names=['psi_' + str(i) for i in range(len(graph.instar[j]))])

        # Add beta variable
        beta_start = psi_start + len(graph.instar[j])
        prob.variables.add(
            obj=[1],
            types=[prob.variables.type.binary],
            names=['beta'])

        # Add alpha variable
        alpha_start = beta_start + 1
        prob.variables.add(
            lb=[0],
            ub=[cplex.infinity],
            types=[prob.variables.type.continuous],
            names=['alpha'])

        # Add weights constraint
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))],
                                       val=[graph.arc_weight_in[j][i] 
                                        for i in range(len(graph.arc_weight_in[j]))])],
            senses=['G'], rhs=[graph.node_threshold[j]],
            names=['weight cover'])

        # Add cover validity constraints. This will modified depending on the 
        # current incumbent.
        prob.linear_constraints.add(senses=['G'], names=['validity_cover'])

        # Add linearization minimum cover weigth
        max_weight_in = max(graph.arc_weight_in[j])
        for i in range(len(graph.instar[j])):
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[psi_start + i] + [alpha_start],
                                           val=[max_weight_in - (1 - max_weight_dev) * graph.arc_weight_in[j][i]] + [1])],
                senses=['L'], rhs=[max_weight_in],
                names=['minimum_' + str(i)])

        # Add cover minimality
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[i for i in range(len(graph.instar[j]))] + [alpha_start],
                                       val=[graph.arc_weight_in[j][i] * (1 - max_weight_dev)
                                        for i in range(len(graph.arc_weight_in[j]))] + [-1])],
            senses=['L'], rhs=[graph.node_threshold[j] + min(thresh_budget, max_thresh_dev * graph.node_threshold[j])],
            names=['minimality'])

        # This is a maximization problem
        prob.objective.set_sense(prob.objective.sense.maximize)

        # Add to the list of problems
        pricing_problems.append(prob)

    return pricing_problems
