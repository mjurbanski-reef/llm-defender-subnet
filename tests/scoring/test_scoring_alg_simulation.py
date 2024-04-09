import bittensor as bt 
from math import log, sqrt
import copy 
import torch
import matplotlib.pyplot as plt

# Score generator
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
    scores = torch.tensor([dist_func(x) for x in x_values])
    print(f"Unweighted scores: {scores}")
    
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
        if float(score) == 0.0:
            normalized_score = 10
        else:
            normalized_score = abs(log(score))
        
        # calculate binned score and update scores
        binned_score = 0.0
        for score_bin in score_bins:
            if score_bin[0] <= normalized_score < score_bin[1]:
                binned_score = score_bin[2]

        scores[i] = binned_score

    print(f"Normalized and binned scores: {scores}")
        
    return scores

def plot_normalize_and_bin_process(dist_func, dist_x_range_low, dist_x_range_high, plot_title, tensor_length = 256):

    unweighted_scores = generate_unweighted_scores(
        dist_func=dist_func, 
        dist_x_range_low=dist_x_range_low, 
        dist_x_range_high=dist_x_range_high, 
        tensor_length=tensor_length)     
    scores = normalize_and_bin(unweighted_scores)
    unweighted_scores, scores = unweighted_scores.tolist(), scores.tolist()
    uids = [x for x, _ in enumerate(scores)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # Plotting on the first subplot
    axes[0].plot(uids, unweighted_scores)
    axes[0].set_title("Current Score Distribution")  # Title for the first subplot
    axes[0].set_xlabel("UID")  # X-axis label for the first subplot
    axes[0].set_ylabel("Score")  # Y-axis label for the first subplot

    # Plotting on the second subplot
    axes[1].plot(uids, scores)
    axes[1].set_title("Binned Score Distribution")  # Title for the second subplot
    axes[1].set_xlabel("UID")  # X-axis label for the second subplot
    axes[1].set_ylabel("Score")  # Y-axis label for the second subplot

    # Adding a title for the entire figure
    fig.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()

def plot_all_normalize_and_bin_processes():

    linear_scores = lambda x: x 
    quadratic_scores = lambda x: x * x
    sqrt_scores = lambda x: sqrt(x)

    lambdas_iterable = [
        linear_scores,
        quadratic_scores,
        sqrt_scores
    ]

    plot_titles = [
        "Effect of Binning on Score Dist: x",
        "Effect of Binning on Score Dist: x^2",
        "Effect of Binning on Score Dist: sqrt(x)"
    ]


    for lambda_func, title in zip(lambdas_iterable, plot_titles):
        plot_normalize_and_bin_process(dist_func = lambda_func, 
                                       dist_x_range_low = 0, 
                                       dist_x_range_high = 1, 
                                       plot_title = title)
        

        
if __name__ == '__main__':

    print("Now testing the normalize & bin process:")

    plot_all_normalize_and_bin_processes()