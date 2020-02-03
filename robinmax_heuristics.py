"""Fast heuristics for robust influence maximization

This module implements fast heuristics for robust influence
maximization.

"""

import numpy as np
import random
import robinmax_auxprob_creator as aux
import time

def random_heuristic(graph, num_seeds, thresh_budget,
                     max_thresh_dev, weight_budget, max_weight_dev,
                     epsilon, debugging, out_f, max_time=600):

    start_time = time.time()

    prob = aux.create_lowerbounding_problem(
        graph, num_seeds, thresh_budget, max_thresh_dev,
        weight_budget, max_weight_dev, epsilon)
    prob.parameters.advance.set(0)

    best_obj = 0

    random_range = range(graph.num_nodes)

    it = 0
    while(time.time() - start_time <= max_time):
        it += 1
        incumbent = [0 for _ in range(graph.num_nodes)]
        random_sample = sorted(random.sample(random_range, max(num_seeds*2, graph.num_nodes)),
                               key=lambda x: len(graph.outstar[x]), reverse=True) 
        for i in range(num_seeds):
            incumbent[random_sample[i]] = 1

        prob.variables.set_lower_bounds(
            [(i, incumbent[i]) for i in range(graph.num_nodes)])

        # Solve lower bounding problem
        prob.set_results_stream(None)
        prob.set_log_stream(None)

        time_left = max(0, (max_time - (time.time() - start_time)))
        prob.parameters.timelimit.set(time_left)
        prob.parameters.threads.set(1)
        prob.solve()

        # Get the optimal solution
        obj = 0
        try:
            obj = prob.solution.get_objective_value()
        except:
            pass

        if (obj > best_obj and prob.solution.get_status() == prob.solution.status.MIP_optimal):
            best_obj = obj
            print('Found new incumbent at itaration {:d} with value: '.format(it),
                obj, file=out_f)

    return best_obj, it

def two_opt_heuristic(graph, num_seeds, thresh_budget,
                      max_thresh_dev, weight_budget, max_weight_dev,
                      epsilon, max_time, debugging, out_f):
    num_nodes = graph.num_nodes
    start_time = time.time()

    prob = aux.create_lowerbounding_problem(
        graph, num_seeds, thresh_budget, max_thresh_dev,
        weight_budget, max_weight_dev, epsilon)
    prob.parameters.advance.set(0)

    best_obj = 0

    it = 0
    while(time.time() - start_time <= max_time):
        print('Restart')
        # Find initial solution
        incumbent = [0 for _ in range(num_nodes)]
        seeds = np.random.choice(num_nodes, num_seeds, replace=False)
        for i in seeds:
            incumbent[i] = 1
        improving = True
        best_obj_this_run = 0
        # Solve lower bounding problem
        prob.variables.set_lower_bounds(
            [(i, incumbent[i]) for i in range(num_nodes)])        
        prob.set_results_stream(None)
        prob.set_log_stream(None)
        time_left = max(0, (max_time - (time.time() - start_time)))
        prob.parameters.timelimit.set(time_left)   
        prob.parameters.threads.set(1)     
        prob.solve()
        it += 1
            
        # Get the optimal solution
        obj = 0
        try:
            obj = prob.solution.get_objective_value()
        except:
            return best_obj, it
            
        if (obj > best_obj and prob.solution.get_status() == prob.solution.status.MIP_optimal):
            best_obj = obj
            print('Found new incumbent at itaration {:d} with value: '.format(it),
                obj, file=out_f)

        if (obj > best_obj_this_run):
            best_obj_this_run = obj

        while improving:
            it += 1
            # Find best 2-swap
            best_new_incumbent = []
            best_new_incumbent_value = obj
            for i in [k for k in range(num_nodes) if incumbent[k] == 1]:
                if (time.time() - start_time > max_time):
                    break
                for j in [k for k in range(num_nodes) if incumbent[k] == 0]:
                    if (time.time() - start_time > max_time):
                        break
                    new_incumbent = [val for val in incumbent]
                    new_incumbent[i] = 0
                    new_incumbent[j] = 1

                    # Solve lower bounding problem
                    prob.variables.set_lower_bounds(
                        [(h, new_incumbent[h]) for h in range(num_nodes)])
                    prob.set_results_stream(None)
                    prob.set_log_stream(None)
                    time_left = max(0, (max_time - (time.time() - start_time)))
                    prob.parameters.timelimit.set(time_left)
                    prob.parameters.threads.set(1)             
                    prob.solve()
                    it += 1            
                    # Get the optimal solution
                    obj = 0
                    try:
                        obj = prob.solution.get_objective_value()
                    except:
                        pass
            
                    if (obj > best_obj and prob.solution.get_status() == prob.solution.status.MIP_optimal):
                        best_obj = obj
                        print('Found new incumbent at iteration {:d} with value: '.format(it),
                              obj, file=out_f)
                    if (obj > best_new_incumbent_value):
                        best_new_incumbent_value = obj
                        best_new_incumbent = [val for val in new_incumbent]
            # -- end for
            if (best_new_incumbent_value > best_obj_this_run):
                improving = True
                best_obj_this_run = best_new_incumbent_value
                incumbent = best_new_incumbent
            else:
                improving = False

    return best_obj, it
