""" Utility function

This module contains some helpful functions.

"""

def least_significant_digit(number):
    """Find out how many digits of precision a number has.

    Parameters
    ----------
    number: float
        The number.
    """
    number_string = str(number)
    if ('.' in number_string):
        return -len(number_string.partition('.')[2])
    else:
        return 0

def epsilon(graph):
    return [(min(10**least_significant_digit(str(weight))
                    for weight in graph.arc_weight_in[i])/2
                if graph.arc_weight_in[i] else 1.0)
               for i in range(graph.num_nodes)]