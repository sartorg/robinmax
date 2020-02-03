"""Callbacks for Branch-and-Cut for robust influence maximization.

This module contains the Cplex callbacks for the Branch-and-Cut
algorithm for robust influence maximization.

"""

import cplex
import cplex.callbacks as cplex_cb
import robinmax_auxprob_creator as aux
import numpy as np
from queue import Queue


class LazyConstraint(object):
    
    def __init__(self, graph, num_seeds, cover_set, thresholds, cover_index,
                 cover_pointer, cover_size, num_covers, thresh_budget,
                 max_thresh_dev, weight_budget, max_weight_dev,
                 lazy_constraints, lazy_info, pruned_info, epsilon,
                 run_as_heuristic, debugging, out_f):
        self.times_called = 0
        self.graph = graph
        self.num_seed = num_seeds
        self.cover_set = cover_set
        self.thresholds = thresholds
        self.cover_index = cover_index
        self.cover_pointer = cover_pointer
        self.cover_size = cover_size
        self.num_covers = num_covers
        self.thresh_budget = thresh_budget
        self.max_thresh_dev = max_thresh_dev
        self.weight_budget = weight_budget
        self.max_weight_dev = max_weight_dev
        self.lazy_constraints = lazy_constraints
        self.rejected_incumbents = set()
        self.accepted_incumbents = set()
        self.seen_incumbents = dict()
        self.lazy_info = lazy_info
        self.pruned_info = pruned_info
        self.seen_pruned_point = set()
        self.run_as_heuristic = run_as_heuristic
        self.debugging = debugging
        self.out_f = out_f
        self.lb_prob = aux.create_lowerbounding_problem(
            graph, num_seeds, thresh_budget, max_thresh_dev,
            weight_budget, max_weight_dev, epsilon)
        self.lb_prob.parameters.advance.set(0)
        # Compute the pointers for phi variables defined in lb_prob
        # It will be handy when updating the thresholds with the optimal
        # values of phi computed in the lb_prob
        self.phi_pointer = dict()
        ind = 0
        for i in range(self.graph.num_nodes):
            for j in self.graph.instar[i]:
                self.phi_pointer[i, j] = ind
                ind += 1
    
    def invoke(self, context):
        self.times_called += 1
        if (context.in_relaxation()):
            # Check if node is about to be pruned
            if (context.get_relaxation_objective() <=
                context.get_double_info(context.info.best_bound)):
                pruned_point = context.get_relaxation_point(
                    0, self.graph.num_nodes-1)
                str_val = ''.join([str(0 if abs(val) <= 1.0e-8 else 1)
                                   for val in pruned_point])
                if (str_val not in self.seen_pruned_point):                
                    self.pruned_info.append(context.get_relaxation_point())
                    self.seen_pruned_point.add(str_val)
            return
        if (not context.is_candidate_point()):
            # This should never happen, but it seems that for
            # numerical reasons sometimes we get an unbounded ray
            context.reject_candidate()
            return
        incumbent = context.get_candidate_point()
        numerical_problem = (sum(abs(val) for val in incumbent) >= 1.0e20)
        if self.debugging:
            print('Incumbent with value {:f} received.'.format(
                context.get_candidate_objective()))
        str_val = ''.join([str(int(round(val)))
                           for val in incumbent[:self.graph.num_nodes]])
        #print(str_val)
        first_time_seen = True
        if (str_val in self.seen_incumbents):
            lazy_constraint, coefficients, lower_bound, theta, phi = self.seen_incumbents[str_val]
            first_time_seen = False
        else:
            self.lb_prob.variables.set_lower_bounds(
                [(i, incumbent[i]) for i in range(self.graph.num_nodes)])
            if (self.debugging):
                self.lb_prob.write('last_lb_problem.lp')
            # Solve lower bounding problem
            self.lb_prob.set_results_stream(None)
            self.lb_prob.set_log_stream(None)
            self.lb_prob.parameters.threads.set(1)
            self.lb_prob.solve()
            if (self.lb_prob.solution.get_status() !=
                self.lb_prob.solution.status.MIP_optimal):
                raise RuntimeError('Could not solve sub-MIP')
            # Get the optimal solution
            lower_bound = self.lb_prob.solution.get_objective_value()
            if self.debugging:
                print('Solved primal problem. Lower bound: ' +
                      str(lower_bound))
                print('incumbent:', str_val)
            theta = self.lb_prob.solution.get_values(self.graph.num_nodes,
                                                     2*self.graph.num_nodes-1)
            if (self.debugging):
                print('Optimal theta for this incumbent:', theta)
            phi = self.lb_prob.solution.get_values(2 * self.graph.num_nodes,
                                                   2 * self.graph.num_nodes + 
                                                   self.graph.num_arcs - 1)
            if (self.debugging):
                print('Optimal phi for this incumbent:', phi)
            # Initialize the coefficients of the cover vars to |S| - 1
            coefficients = [val - 1 for val in self.cover_size]
            # Find the invalid cover and update their coefficient to |S|
            invalid_covers = []
            for i in range(self.num_covers):
                node, index = self.cover_pointer[i]
                # Get the sum of the optimal phi associated with this cover
                phi_cover_sum = sum(phi[self.phi_pointer[node, j]] 
                            for j in self.cover_set[node][index])
                curr_thresh = self.graph.node_threshold[node] + theta[node] + phi_cover_sum
                if (#curr_thresh <= self.thresholds[node][index][0] or
                    curr_thresh > self.thresholds[node][index][1]):
                    coefficients[i] = coefficients[i] + 1
                    invalid_covers.append(i)
            if (self.debugging):
                print('Inactive covers:', invalid_covers)
            # Create constraint
            lazy_constraint = [cplex.SparsePair(
                ind=[i for i in range(self.graph.num_nodes,
                                      2*self.graph.num_nodes +
                                      self.num_covers + 1)],
                val=coefficients + [1] * self.graph.num_nodes + [-1])]
            # Save the constraint info 
            self.seen_incumbents[str_val] = (lazy_constraint, coefficients, lower_bound, theta, phi)
            self.lazy_info.append((incumbent, theta, phi))

        # Compute constraint value
        lhs = sum(incumbent[self.graph.num_nodes + i]*coefficients[i]
                for i in range(self.num_covers))
        lhs += sum(incumbent[self.graph.num_nodes + self.num_covers + i]
                for i in range(self.graph.num_nodes))
        if (incumbent[-1] > lower_bound and lhs > incumbent[-1] + 1.0e-8 and
            not numerical_problem):
            raise RuntimeError('Incumbent and lower bound differ,' +
                               ' but no cut has been added')
        # Add lazy constraint if it is violated
        if (lhs <= incumbent[-1] - 1.0e-8 or first_time_seen or
            numerical_problem):
            if (not str_val in self.rejected_incumbents):
                self.lazy_constraints.append(lazy_constraint)
            if (self.debugging):
                print('lhs', lhs, 'z', incumbent[-1])
            if (self.debugging):
                print('Incumbent rejected.')
            context.reject_candidate(
                constraints=lazy_constraint,
                senses=['G'], rhs=[0.0])
            self.rejected_incumbents.add(str_val)
        else:
            # We found a new incumbent
            if (str_val not in self.accepted_incumbents):
                if (self.debugging or not self.run_as_heuristic):
                    print('{:8d}   {:9d}   {:9f}'.format(
                        self.times_called, 
                        len(self.rejected_incumbents),
                        context.get_candidate_objective()),
                          file=self.out_f,
                          flush=True)
                self.accepted_incumbents.add(str_val)

class LazyConstraint2(object):
    
    def __init__(self, graph, num_seeds, cover_set, thresholds, cover_index,
                 cover_pointer, cover_size, num_covers, thresh_budget,
                 max_thresh_dev, weight_budget, max_weight_dev,
                 lazy_constraints, epsilon,run_as_heuristic, cutoff,
                 debugging, out_f):
        self.times_called = 0
        self.graph = graph
        self.num_seed = num_seeds
        self.cover_set = cover_set
        self.thresholds = thresholds
        self.cover_index = cover_index
        self.cover_pointer = cover_pointer
        self.cover_size = cover_size
        self.num_covers = num_covers
        self.thresh_budget = thresh_budget
        self.max_thresh_dev = max_thresh_dev
        self.weight_budget = weight_budget
        self.max_weight_dev = max_weight_dev
        self.lazy_constraints = lazy_constraints
        self.num_rejected_incumbents = 0
        self.seen_incumbents = dict()
        self.run_as_heuristic = run_as_heuristic
        self.cutoff = cutoff
        self.lazy_obj = list()
        self.debugging = debugging
        self.out_f = out_f
        self.lb_prob = aux.create_lowerbounding_problem(
            graph, num_seeds, thresh_budget, max_thresh_dev,
            weight_budget, max_weight_dev, epsilon)
        self.best_sol_value = cutoff
        # Compute the pointers for phi variables defined in lb_prob
        # It will be handy when updating the thresholds with the optimal
        # values of phi computed in the lb_prob
        self.phi_pointer = dict()
        ind = 0
        for i in range(self.graph.num_nodes):
            for j in self.graph.instar[i]:
                self.phi_pointer[i, j] = ind
                ind += 1

    def solve_lb_problem(self, incumbent):
        self.lb_prob.variables.set_lower_bounds(
            [(i, incumbent[i]) for i in range(self.graph.num_nodes)])
        if (self.debugging):
            self.lb_prob.write('last_lb_problem.lp')
        # Solve lower bounding problem
        self.lb_prob.set_results_stream(None)
        self.lb_prob.set_log_stream(None)
        self.lb_prob.parameters.threads.set(1)
        self.lb_prob.solve()
        if (self.lb_prob.solution.get_status() !=
            self.lb_prob.solution.status.MIP_optimal):
            raise RuntimeError('Could not solve sub-MIP')
        # Get the optimal solution
        lower_bound = self.lb_prob.solution.get_objective_value()
        theta = np.asarray(
            self.lb_prob.solution.get_values(self.graph.num_nodes,
                                             2*self.graph.num_nodes-1))
        phi = np.asarray(
            self.lb_prob.solution.get_values(2 * self.graph.num_nodes,
                                             2 * self.graph.num_nodes + 
                                             self.graph.num_arcs - 1))
        return lower_bound, theta, phi
    
    def invoke(self, context):
        self.times_called += 1
        if (not context.is_candidate_point()):
            # This should never happen, but it seems that for
            # numerical reasons sometimes we get an unbounded ray
            context.reject_candidate()
            return
        candidate_point = context.get_candidate_point()
        # We are actually only interested in the y values
        incumbent = np.asarray(candidate_point[:self.graph.num_nodes])
        if self.debugging:
            print('Incumbent with value {:f} received.'.format(
                context.get_candidate_objective()))
        str_val = ''.join([str(int(round(val))) for val in incumbent])

        if (str_val in self.seen_incumbents):
            lower_bound = self.seen_incumbents[str_val]
            theta, phi = None, None
        else:
            lower_bound, theta, phi = self.solve_lb_problem(incumbent)
            if self.debugging:
                print('Solved primal problem. Lower bound: ' +
                      str(lower_bound))
                print('incumbent:', str_val)
                print('Optimal theta for this incumbent:', theta)
                print('Optimal phi for this incumbent:', phi)
            # Initialize the coefficients of the cover vars to |S| - 1
            coefficients = [val - 1 for val in self.cover_size]
            # Find the invalid cover and update their coefficient to |S|
            invalid_covers = []
            for i in range(self.num_covers):
                node, index = self.cover_pointer[i]
                # Get the sum of the optimal phi associated with this cover
                phi_cover_sum = sum(phi[self.phi_pointer[node, j]] 
                            for j in self.cover_set[node][index])
                curr_thresh = self.graph.node_threshold[node] + theta[node] + phi_cover_sum
                if (#curr_thresh <= self.thresholds[node][index][0] or
                    curr_thresh > self.thresholds[node][index][1]):
                    coefficients[i] = coefficients[i] + 1
                    invalid_covers.append(i)
            if (self.debugging):
                print('Inactive covers:', invalid_covers)
            # Create constraint
            ind = np.asarray([i for i in range(self.graph.num_nodes,
                                               2*self.graph.num_nodes +
                                               self.num_covers)])
            val = np.asarray([1] * self.graph.num_nodes + coefficients)
            lazy_constraint = (ind, val)
            self.seen_incumbents[str_val] = lower_bound

        if (lower_bound <= self.best_sol_value - 1.0e-8):
            if (self.debugging):
                print('Incumbent rejected.')
            context.reject_candidate()
            self.num_rejected_incumbents += 1
            #self.rejected_incumbents.add(str_val)
        elif (abs(lower_bound - self.best_sol_value) <= 1.0e-8):
            if (context.get_candidate_objective() <= self.best_sol_value +
                1.0e-8):
                # This solution matches the best known value: we can
                # accept it
                if (self.debugging or not self.run_as_heuristic):
                    print('{:8d}   {:9d}   {:9f}'.format(
                        self.times_called, 
                        self.num_rejected_incumbents,
                        lower_bound),
                          file=self.out_f)
                self.best_sol_value = lower_bound
                self.best_solution = incumbent
            else:
                # This is a different solution but with same lower
                # bound: reject
                context.reject_candidate()
                self.num_rejected_incumbents += 1
        else:
            # This is a better solution
            self.lazy_constraints.append(lazy_constraint)
            if (theta is None):
                lower_bound, theta, phi = self.solve_lb_problem(incumbent)
            self.lazy_obj = [lazy_constraint, (incumbent, theta, phi)]
            if (self.debugging or not self.run_as_heuristic):
                print('{:8d}   {:9d}   {:9f}'.format(
                    self.times_called, 
                    self.num_rejected_incumbents,
                    lower_bound),
                      file=self.out_f)
            self.best_sol_value = lower_bound
            self.best_solution = incumbent
            if (context.get_int_info(context.info.node_count) <= 100000 or
                (self.best_sol_value > self.cutoff*1.05 and 
                 context.get_double_info(context.info.best_bound) >=
                 self.graph.num_nodes - 1 and
                 context.get_int_info(context.info.node_count) <= 1000000)):
                context.abort()


class HeuristicSolution(object):
    
    def __init__(self, graph, num_seeds, thresh_budget, max_thresh_dev,
                 weight_budget, max_weight_dev, debugging):
        self.times_called = 0
        self.graph = graph
        self.num_seed = num_seeds
        self.thresh_budget = thresh_budget
        self.max_thresh_dev = max_thresh_dev
        self.weight_budget = weight_budget
        self.max_weight_dev = max_weight_dev
        self.debugging = debugging
        self.lb_prob = aux.create_lowerbounding_problem(
            graph, num_seeds, thresh_budget, max_thresh_dev,
            weight_budget, max_weight_dev)
        self.lb_prob.parameters.advance.set(0)
        self.best_lower_bound = float('-inf')
    
    def invoke(self, context):
        self.times_called += 1
        incumbent = context.get_candidate_point(0, self.graph.num_nodes - 1)
        if self.debugging:
            print('Incumbent with value {:f} received.'.format(
                context.get_candidate_objective()))
            print('Incumbent:', context.get_candidate_point())

        self.lb_prob.variables.set_lower_bounds(
            [(i, incumbent[i]) for i in range(self.graph.num_nodes)])
        self.lb_prob.set_results_stream(None)
        self.lb_prob.set_log_stream(None)
        self.lb_prob.parameters.threads.set(1)
        self.lb_prob.solve()
        lower_bound = self.lb_prob.solution.get_objective_value()
        if self.debugging:
            print('Solved primal problem. Lower bound: ' + str(lower_bound))
        if (lower_bound > self.best_lower_bound):
            if self.debugging:
                print('*** Found new lower bound!!! ***')
            self.best_lower_bound = lower_bound
            self.best_solution = self.lb_prob.solution.get_values()
        # If the candidate is worse than the best known solution, we
        # can still accept it to make Cplex happy; otherwise we
        # reject.
        if (context.get_candidate_objective() > self.best_lower_bound):
            context.reject_candidate()
        # Check dual bound. If we solved the problem, abort.
        if (context.get_double_info(context.info.best_bound) <=
            self.best_lower_bound):
            context.abort()



class TightenBounds(cplex_cb.UserCutCallback):
    def __call__(self):
        # Check depth. The code 222 is the magic code to obtain the
        # depth of a node -- might change with different versions.
        depth = self._get_node_info(222, 0)
        if (depth % self.frequency == 0 and self.is_after_cut_loop()):
            # Adjust objective function bound
            lpsol = self.get_values()
            obj = self.get_objective_value()
            prob = self.bt_prob
            prob.linear_constraints.set_linear_components(
                prob.linear_constraints.get_num() - 1,
                [[self.num_covers + i for i in range(self.graph.num_nodes)],
                 lpsol[:self.graph.num_nodes]])
            prob.linear_constraints.set_rhs(
                prob.linear_constraints.get_num() - 1, obj)
            w_start = 2*self.graph.num_nodes + self.num_covers
            for i in range(self.graph.num_nodes):
                prob.objective.set_linear(
                    [(self.num_covers + j, 1 if i == j else 0)
                     for j in range(self.graph.num_nodes)])
                prob.parameters.threads.set(1)
                prob.solve()
                if (prob.solution.get_status() ==
                    prob.solution.status.optimal):
                    mu_bound = prob.solution.get_objective_value()
                    print('** Bound on mu ', i, mu_bound)
                    self.add_local(
                        cplex.SparsePair(ind = [i, w_start + i],
                                         val = [-mu_bound, 1]), 'L', 0)
                    self.add_local(
                        cplex.SparsePair(ind = [self.num_covers + i],
                                         val = [1]), 'L', mu_bound)
            self.abort_cut_loop()
            
            
class BranchingRule(cplex_cb.BranchCallback):

    def __call__(self):
        depth = self._get_node_info(222, 0)
        if (depth % 1 == 0 and depth < 20):
            y = self.get_values(0, self.graph.num_nodes - 1)
            vec = np.array(y)
            nz = np.nonzero(np.abs(y) > 1.0e-5)[0]
            sorted_vec = np.argsort(vec[nz])
            # Find a good sum of fractional values
            acc = 0
            j = 0
            while (j < len(sorted_vec) and
                   ((acc <= 0.5) or (abs(acc - round(acc)) <= 1.0e-1))):
                acc += vec[nz][sorted_vec[j]]
                j += 1
            if (j == len(sorted_vec)):
                for i in range(self.get_num_branches()):
                    self.make_cplex_branch(i)
            else:
                branch_indices = [int(i) for i in nz[sorted_vec[:(j+1)]]]
                branch_value = sum(vec[branch_indices])
                self.make_branch(
                    self.get_objective_value(),
                    constraints=[([branch_indices,
                                   [1] * len(branch_indices)],
                                  "L", float(np.floor(branch_value)))
                    ])
                self.make_branch(
                    self.get_objective_value(),
                    constraints=[([branch_indices,
                                   [1] * len(branch_indices)],
                                  "G", float(np.ceil(branch_value)))
                    ])
        else:
            for i in range(self.get_num_branches()):
                self.make_cplex_branch(i)
        
