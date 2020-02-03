"""Robust influence maximization.

This module implements an algorithm for robust influence
maximiztion.

"""

import sys
import argparse
import ast
import numpy as np

import robinmax_bac_indicator as bac
import robinmax_graph as gr
import robinmax_cover_generator as cg
import robinmax_utils as util
import robinmax_heuristics as heurs
import robinmax_column_generation as col
import time

def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.
    """
    parser.add_argument('graph_file', action = 'store', type = str,
                        help = 'File containing graph connectivity information.')
    parser.add_argument('--debug', action = 'store', dest = 'debug',
                        default = False, type = ast.literal_eval,
                        help = 'Print debug info. Default False.')
    parser.add_argument('--robust_thresh_budget', '-rtb', type = float,
                        default = 0.0, help = 'Node threshold budget robustness.')
    parser.add_argument('--max_thresh_dev', '-td', type = float,
                        default = 0.0, help = 'Maximum node threshold deviation.')
    parser.add_argument('--robust_weight_budget', '-rwb', type = float,
                        default = 0.0, help = 'Arc weight budget robustness.')
    parser.add_argument('--max_weight_dev', '-wd', type = float,
                        default = 0.0, help = 'Maximum arc weight deviation.')
    parser.add_argument('--max_cover_size', '-cs', type = int,
                        default = -1, help = 'Maximum size for the ' +
                        'activation covers generated. Default -1,' +
                        ' in which case it is the number of nodes.')
    parser.add_argument('--time', '-t', type = float,
                        default = 3600,
                        help = 'Max time. Default 3600.')
    parser.add_argument('--num_seeds', '-ns', type = int,
                        default = 3, help = 'Number of seed nodes. ' +
                        'Default 3.')
    parser.add_argument('--heuristics', '-heurs', action='store',
                        dest='heuristics',
                        default=-1, type=int,
                        help='Solve the problem using heuristics.' + 
                            ' 1 = Column generation, 2 = Random,' +
                            ' 3 = Two-opt, 4 = Old column generation. ' +
                            ' Default -1, no heuristic.')
    parser.add_argument('--disable_cuts', '-dc', action='store',
                        dest='disable_cuts',
                        default=False, type=ast.literal_eval,
                        help='Disable CPLEX cuts. Default False.')
    parser.add_argument('--cg_init_iters', '-cgi', action='store',
                        dest='cg_init_iters', default=20, type=int,
                        help='Number of CG initialization interations. ' +
                        'Default 20.')
    parser.add_argument('--max_columns_per_round', '-mcr', action='store',
                        dest='max_columns_per_round', default=10000,
                        type=int, help='Stop generating columns after ' +
                        'this many. Default 10000.')
    parser.add_argument('--max_col_iters_per_round', '-mci', action='store',
                        dest='max_col_iters_per_round', default=0,
                        type=int, help='Stop trying to generate columns after ' +
                        'this many iterations. Default 0.')
    parser.add_argument('--max_pricing_iters', '-mpi', action='store',
                        dest='max_pricing_iters', default=0,
                        type=int, help='Stop solving pricing problems after ' +
                        'this many. Default 0.')
    parser.add_argument('--num_init_covers', '-nic', action='store',
                        dest='num_init_covers', default=5000,
                        type=int, help='The initial number of generated ' +
                        'covers. Default 5000.')
    parser.add_argument('--random_seed', '-rs', action='store',
                        dest='random_seed', default=1981231712,
                        type=int, help='Random seed. Default 1981231712.')
    parser.add_argument('--lp', '-lp', action='store',
                        dest='lp',
                        default=False, type=ast.literal_eval,
                        help='Solve the relaxed problem. Default False.')

# -- end function

def robinmax(graph, num_seeds, max_cover_size, thresh_budget=0,
             max_thresh_dev=0.0, weight_budget=0.0,
             max_weight_dev=0.0, max_time=3600, 
             heuristics=-1, cg_init_iters=20, max_columns_per_round=10000,
             max_col_iters_per_round=0, max_pricing_iters=0,
             num_init_covers=5000, debugging=False, disable_cuts=False, 
             lp=False, out_f=sys.__stdout__):

    # Compute the epsilon to use throughout the algorithm
    epsilon = util.epsilon(graph)

    # Print info
    print('\nGRAPH')
    print('Name: {:s}'.format(graph.name))
    print('Nodes: {:d}'.format(graph.num_nodes))
    print('Arcs: {:d}'.format(graph.num_arcs))
    print('')
    print('PARAMETERS')
    print('Seeds: {:d}'.format(int(num_seeds)))
    print('Cover size: {:d}'.format(int(max_cover_size)))
    print('Robustness threshold budget: {:.2f}'.format(thresh_budget))
    print('Max threshold deviation: {:.2f}'.format(max_thresh_dev))
    print('Robustness weight budget: {:.2f}'.format(weight_budget))
    print('Max weight deviation: {:.2f}'.format(max_weight_dev))
    print('Time limit: {:.1f}'.format(max_time))
    print('Disable cuts: {:s}'.format(str(disable_cuts)))
    print('Solve as LP: {:s}'.format(str(lp)))
    print('Epsilon: {:.2e}'.format(np.mean(epsilon)))
    print('Debugging: {:s}'.format(str(debugging)))
    print('Heuristics: {:d}'.format(heuristics))
    print('Max columns per round: {:d}'.format(max_columns_per_round))
    print('Max column iters per round: {:d}'.format(max_col_iters_per_round))
    print('Max pricing iters: {:d}'.format(max_pricing_iters))
    print('Number initial covers: {:d}'.format(num_init_covers))
    print('Output file: {:s}'.format(str(out_f.name)))
    print('')

    str_args = ';'.join(['Name', graph.name, 'Nodes', str(graph.num_nodes),
        'Arcs', str(graph.num_arcs), 'Seeds', str(num_seeds),
        'Max cover size', str(max_cover_size),
        'Robustness threshold budget', str(thresh_budget),
        'max threshold deviation', str(max_thresh_dev),
        'Robustness weight budget', str(weight_budget),
        'Max weigh deviation', str(max_weight_dev),
        'Time limit', str(max_time), 
        'Disable cuts', str(disable_cuts),
        'Solve as LP', str(lp), 'Epsilon', str(np.mean(epsilon)),
        'Debugging', str(debugging), 'Heuristics', str(heuristics)])

    try:

        if (heuristics == 2):
            start_time = time.time()

            best_obj, it = heurs.random_heuristic(graph, num_seeds, thresh_budget,
                            max_thresh_dev, weight_budget, max_weight_dev,
                            epsilon, debugging, out_f, max_time)

            str_results = ';'.join(['Elapsed time', str(time.time() - start_time),
                'Iterations', str(it), 'Best objective', str(best_obj)])
        elif (heuristics == 3):
            start_time = time.time()

            best_obj, iterations = heurs.two_opt_heuristic(
                graph, num_seeds, thresh_budget,
                max_thresh_dev, weight_budget, max_weight_dev,
                epsilon, max_time, debugging, out_f)

            str_results = ';'.join([
                'Elapsed time', str(time.time() - start_time),
                'Iterations', str(iterations),
                'Best objective', str(best_obj)])
        
        elif (heuristics == 4):
            start_time = time.time()
            previous_num_covers = -1
            best_obj = 0
            covers = [list() for _ in range(graph.num_nodes)]
            thresholds = [list() for _ in range(graph.num_nodes)]
            num_covers = 0
            iter_number = 0
            stop = False
            to_exclude = list()
            while (not stop and
                   time.time() - start_time <= max_time and
                   iter_number < cg_init_iters):

                time_left = max(0, (max_time - (time.time() - start_time)))

                lazy_info, pruned_info, results = bac.robinmax_bac(
                    graph, num_seeds, max_cover_size, thresh_budget,
                    max_thresh_dev, weight_budget, max_weight_dev,
                    max_time=time_left, epsilon=epsilon, debugging=debugging,
                    disable_cuts=disable_cuts, lp=lp, covers=covers, 
                    thresholds=thresholds,
                    save_pruned=(num_covers==previous_num_covers),
                    run_as_heuristic=True, num_nodes=5000, 
                    points_to_exclude=to_exclude, out_f=out_f)

                print('ITERATION: ', iter_number)
                print('Integer solutions: {:d}'.format(len(lazy_info)))
                print('Pruned points: {:d}'.format(len(pruned_info)))

                time_left = max(0, (max_time - (time.time() - start_time)))

                # Generate covers through column generation
                num_generated_covers, obj = col.column_generation(
                    graph, lazy_info, pruned_info, covers, thresholds,
                    thresh_budget, max_thresh_dev, weight_budget,
                    max_weight_dev, time_left, epsilon, debugging, out_f)

                if (round(best_obj) < round(obj)):
                    best_obj = obj 
                print('Best objective: {:.2f}'.format(best_obj))
                print('Generated covers (#): ', num_generated_covers)

                if (previous_num_covers == num_covers and
                    num_generated_covers == 0):
                    stop = True

                to_exclude.extend([incumbent for (incumbent, theta, phi)
                                in lazy_info])

                previous_num_covers = num_covers
                num_covers += num_generated_covers
                iter_number += 1

            if (time.time() - start_time <= max_time):
                # Run once more with all columns found so far
                time_left = max(0, (max_time - (time.time() - start_time)))
                lazy_info, pruned_info, results = bac.robinmax_bac(
                    graph, num_seeds, max_cover_size, thresh_budget,
                    max_thresh_dev, weight_budget, max_weight_dev,
                    max_time=time_left, epsilon=epsilon, debugging=debugging,
                    disable_cuts=disable_cuts, lp=lp, covers=covers, 
                    thresholds=thresholds, save_pruned=False,
                    run_as_heuristic=False, out_f=out_f)
                # Update best obj
                if (results[4] > best_obj):
                    best_obj = results[4]
            
            str_results = ';'.join(['Elapsed time', str(time.time() - start_time),
                'Iterations', str(iter_number), 'Generated covers', str(num_covers),
                'Best objective', str(best_obj)])

        elif (heuristics == 1):
            start_time = time.time()

            covers = [list() for _ in range(graph.num_nodes)]
            thresholds = [list() for _ in range(graph.num_nodes)]

            max_size_covers = 2
            num_covers = sum([len(c) for c in covers])
            while (num_covers <= num_init_covers and
                   max_size_covers < graph.num_nodes):
                max_size_covers += 1
                covers, thresholds = cg.generate_minimal_covers(
                    graph, max_size_covers, thresh_budget, max_thresh_dev,
                    weight_budget, max_weight_dev)
                new_num_covers = sum([len(c) for c in covers])
                if (new_num_covers == num_covers and new_num_covers > 0):
                    max_size_covers = graph.num_nodes
                num_covers = new_num_covers

            time_left = max(0, (max_time - (time.time() - start_time)))

            results = bac.robinmax_bac_restart(
                graph, num_seeds, max_cover_size, thresh_budget,
                max_thresh_dev, weight_budget, max_weight_dev,
                time_left, epsilon, debugging, disable_cuts, lp,
                covers=covers, thresholds=thresholds, save_pruned=False,
                run_as_heuristic=True, cg_init_iters=cg_init_iters,
                max_columns_per_round=max_columns_per_round, 
                max_pricing_iters=max_pricing_iters,
                max_col_iters_per_round=max_col_iters_per_round, out_f=out_f)

            str_results = ';'.join(['Elapsed time', str(time.time() - start_time),
                'Iterations', str(results[11]), 'Generated covers', 
                str(sum([len(covers[i]) for i in range(graph.num_nodes)])),
                'Best bound', str(results[3]),
                'Best objective', str(results[4])])

        else:
            start_time = time.time()

            covers, thresholds = cg.generate_minimal_covers(
                graph, max_cover_size, thresh_budget, max_thresh_dev,
                weight_budget, max_weight_dev)

            time_left = max(0, (max_time - (time.time() - start_time)))

            results = bac.robinmax_bac_restart(
                graph, num_seeds, max_cover_size, thresh_budget,
                max_thresh_dev, weight_budget, max_weight_dev,
                time_left, epsilon, debugging, disable_cuts, lp, covers=covers, 
                thresholds=thresholds, save_pruned=False,
                run_as_heuristic=False, out_f=out_f)

            print('Cover time (s): {:.2f}'.format(max_time - time_left), file=out_f)
            print('')

            str_results = ';'.join(['Elapsed time', str(time.time() - start_time),
                'CPLEX time', str(results[0]), 
                'Cover time', str(max_time - time_left), 
                'Nodes (#)', str(results[1]), 'Gap (%)', str(results[2]),
                'Best bound', str(results[3]), 'Best objective', str(results[4]),
                'Covers (#)', str(results[5]), 'Lazy cuts', str(results[6]),
                'Nonzero theta at optimum (#)', str(results[7]),
                'Max theta at optimum', str(results[8]),
                'Nonzero phi at optimum (#)', str(results[9]),
                'Max theta at optimum', str(results[10])])
    
    except Exception  as e:
        print('Problem with graph: {:s}. \n'.format(graph.name) + str(e))
        str_results = '{:s}'.format(str(e))
        raise
    
    print("")
    print(str_args + ';' +  str_results, flush=True)
    

    return

# -- end function

if (__name__ == '__main__'):
    if (sys.version_info[0] < 3):
        print('Error: this software requires Python 3 or later')
        exit()
    parser = argparse.ArgumentParser(description = 'Branch-and-Cut for ' +
                                     'robust influence maximization.')
    # Add options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    graph = gr.read_text_graph(args.graph_file)

    # Setting default value of max_cover_size
    max_cover_size = args.max_cover_size
    if (max_cover_size == -1):
        max_cover_size = graph.num_nodes

    if args.heuristics not in [-1, 1, 2, 3]:
            print('Invalid value for heuristics parameter.' + 
            ' Check python3 robinmax.py --help.')
            exit()

    robinmax(graph, args.num_seeds, max_cover_size,
             args.robust_thresh_budget, args.max_thresh_dev,
             args.robust_weight_budget, args.max_weight_dev,
             args.time, args.heuristics, args.cg_init_iters,
             args.max_columns_per_round, args.max_col_iters_per_round,
             args.max_pricing_iters, args.num_init_covers, args.debug,
             args.disable_cuts, args.lp)
    #y_names = ['y_' + str(i) for i in range(graph.num_nodes)]
    #for i, name in enumerate(y_names):
    #    print(name, data.best_incumbent[i])
