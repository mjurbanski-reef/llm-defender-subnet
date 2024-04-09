import bittensor as bt 
from math import log 
import copy 
import torch
import pandas as pd 
import seaborn as sns 

def generate_unweighted_scores(dist_func, dist_x_range_low, dist_x_range_high, tensor_length=256):
    """
    Generates a torch tensor of size tensor_length using lambda function dist_func, 
    using values found in the range starting from dist_x_range_low and ending at dist_x_range_high.

    Args:
    - dist_func (function): A lambda function or any callable that takes a single argument.
    - dist_x_range_low (float): The lower bound of the x values to be used.
    - dist_x_range_high (float): The upper bound of the x values to be used.
    - tensor_length (int, optional): The length of the tensor to be generated. Default is 256.

    Returns:
    - torch.Tensor: A tensor of size tensor_length filled with values computed by dist_func.
    """
    
    # Generate a range of x values from dist_x_range_low to dist_x_range_high
    x_values = torch.linspace(dist_x_range_low, dist_x_range_high, tensor_length)
    
    # Apply the distribution function to each value in x_values
    scores = dist_func(x_values)
    
    return scores

def normalize_and_bin(unweighted_scores):
    """Normalizes miner scores according to a distribution and then bins them"""

    scores = copy.deepcopy(unweighted_scores)

    score_bins = [ #[range_low, range_high, binned_score]
        [0, 0.03, 1],
        [0.03, 0.11, 0.9],
        [0.11, 0.22, 0.8],
        [0.22, 0.35, 0.7],
        [0.35, 0.51, 0.6],
        [0.51, 0.69, 0.5],
        [0.69, 0.91, 0.4],
        [0.91, 1.2, 0.3],
        [1.2, 1.6, 0.2],
        [1.6, 2.3, 0.1]
    ]

    # iterate through scores and turn each into a binned_score
    for i, score in enumerate(unweighted_scores):

        # calculate normalized score
        normalized_score = abs(log(score))
        
        # calculate binned score and update scores
        binned_score = 0.0
        for score_bin in score_bins:
            if score_bin[0] <= normalized_score < score_bin[1]:
                binned_score = score_bin[2]

        scores[i] = binned_score
        
    return scores
                
