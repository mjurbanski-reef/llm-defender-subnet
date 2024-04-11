from math import log, sqrt
import copy 
import torch
import matplotlib.pyplot as plt
import bittensor as bt 
import argparse 
import random

# query miner weights from metagraph 
def query_miner_weights_from_metagraph():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=14)
    parser.add_argument("--subtensor.network", type=str, default='finney')
    config = bt.config(parser=parser)
    subtensor = bt.subtensor(config=config, network = 'finney')
    metagraph = subtensor.metagraph(config.netuid, lite=False)
    metagraph.sync(subtensor=subtensor)
    # modify this line depending on what you want to query (B,C,D,E,I,R,S,T,Tv,W)
    query = metagraph.W
    for q in query:
        print(q)
    return query

def normalize_list(weights):

    max_weight = max(weights)
    normalized_weights = []
    for weight in weights: 
        if weight != 0.0 or weight != 0:
            normalized_weights.append((weight / max_weight))
    return torch.tensor(sorted(normalized_weights))

def get_static_miner_weights_dist():
    weights = [
        0.0042, 0.0042, 0.0042, 0.0042, 0.0000, 0.0042, 0.0000, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0042, 0.0042, 0.0042, 0.0043,
        0.0043, 0.0016, 0.0043, 0.0000, 0.0043, 0.0043, 0.0000, 0.0044, 0.0043,
        0.0043, 0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043,
        0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0039, 0.0042, 0.0043,
        0.0000, 0.0042, 0.0043, 0.0000, 0.0042, 0.0043, 0.0043, 0.0042, 0.0042,
        0.0043, 0.0000, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0000,
        0.0043, 0.0043, 0.0043, 0.0000, 0.0044, 0.0000, 0.0043, 0.0043, 0.0043,
        0.0000, 0.0043, 0.0044, 0.0044, 0.0044, 0.0042, 0.0042, 0.0042, 0.0044,
        0.0042, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0000, 0.0043, 0.0039,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043,
        0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0043,
        0.0042, 0.0043, 0.0042, 0.0043, 0.0042, 0.0043, 0.0000, 0.0039, 0.0043,
        0.0043, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043, 0.0039, 0.0043, 0.0043,
        0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0042, 0.0043, 0.0043,
        0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0000, 0.0043, 0.0045, 0.0000,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0042, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0042,
        0.0043, 0.0042, 0.0043, 0.0000, 0.0043, 0.0000, 0.0043, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0042, 0.0043, 0.0000, 0.0043,
        0.0043, 0.0043, 0.0000, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0000, 0.0000, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0043,
        0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0042, 0.0043, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0045, 0.0043,
        0.0039, 0.0042, 0.0043, 0.0043, 0.0043, 0.0043, 0.0045, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043,
        0.0042, 0.0043, 0.0043, 0.0043
    ]

    return torch.tensor(normalize_list(weights))

# abs(ln(x)) normalization and binning
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
        "Effect of Normalization & Binning on Score Dist: x",
        "Effect of Normalization & Binning on Score Dist: x^2",
        "Effect of Normalization & Binning on Score Dist: sqrt(x)"
    ]

    range_lows = [
        0.3,
        0.5,
        0.0
    ]

    range_highs = [
        1.0,
        1.0,
        1.0
    ]

    for lambda_func, title, range_low, range_high in zip(lambdas_iterable, plot_titles, range_lows, range_highs):
        plot_normalize_and_bin_process(dist_func = lambda_func, 
                                       dist_x_range_low = range_low, 
                                       dist_x_range_high = range_high, 
                                       plot_title = title)

# multi-dimensional averages and binning
def generate_scrambled_unweighted_scores(dist_func, dist_x_range_low, dist_x_range_high, tensor_length=256):
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

    # Generate a random permutation of indices
    permuted_indices = torch.randperm(tensor_length)

    # Use the permuted indices to scramble the scores
    scrambled_scores = scores[permuted_indices]
    
    return scrambled_scores.tolist()
        
def get_unweighted_scores_per_analyzer(dist_func, dist_x_range_low, dist_x_range_high, tensor_length=256):

    analyzers = [
        'Prompt Injection',
        'Sensitive Information',
        'Third Analyzer',
        'Fourth Analyzer'
    ]

    scores_dict = {}

    for analyzer in analyzers:

        scores_dict[analyzer] = generate_scrambled_unweighted_scores(dist_func, dist_x_range_low, dist_x_range_high, tensor_length=tensor_length)

    return scores_dict
        
def n_dim_binning(scores_dict):

    avgs_for_all_uids = []

    for i, _ in enumerate(scores_dict['Prompt Injection']):

        avgs = []

        for analyzer1 in scores_dict:
            for analyzer2 in scores_dict:
                if analyzer1 != analyzer2:
                    avg1, avg2 = scores_dict[analyzer1][i], scores_dict[analyzer2][i]
                    avgs.append(((avg1 + avg2) / 2))

        avgs_for_all_uids.append(avgs)

    avg_for_all_uids = []

    for uid_avgs in avgs_for_all_uids:
        avg_for_all_uids.append((sum(uid_avgs) / len(uid_avgs)))

    score_bins = [ #[range_low, range_high, binned_score]
        [0.95, 1.0, 1.0],
        [0.88, 0.95, 0.9],
        [0.79, 0.88, 0.8],
        [0.69, 0.79, 0.7],
        [0.60, 0.69, 0.6],
        [0.52, 0.60, 0.5],
        [0.45, 0.52, 0.4],
        [0.30, 0.45, 0.3],
        [0.15, 0.30, 0.2],
        [0.0, 0.15, 0.1]
    ]

    scores = []

    # iterate through scores and turn each into a binned_score
    for i, score in enumerate(avg_for_all_uids):

        # calculate binned score and update scores
        binned_score = 0.0
        for score_bin in score_bins:
            if score_bin[0] < score <= score_bin[1]:
                binned_score = score_bin[2]

        scores.append(binned_score)

    return scores, avg_for_all_uids

def plot_n_dim_binning_process(dist_func, dist_x_range_low, dist_x_range_high, plot_title):

    scores_dict = get_unweighted_scores_per_analyzer(dist_func, dist_x_range_low, dist_x_range_high)
    scores, unweighted_scores = n_dim_binning(scores_dict)
    new_unweighted_scores = sorted(unweighted_scores)
    new_scores = sorted(scores)
    uids = [x for x, _ in enumerate(scores)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # Plotting on the first subplot
    axes[0].plot(uids, new_unweighted_scores)
    axes[0].set_title("Current Score Distribution")  # Title for the first subplot
    axes[0].set_xlabel("UID")  # X-axis label for the first subplot
    axes[0].set_ylabel("Score")  # Y-axis label for the first subplot

    # Plotting on the second subplot
    axes[1].plot(uids, new_scores)
    axes[1].set_title("Binned Score Distribution")  # Title for the second subplot
    axes[1].set_xlabel("UID")  # X-axis label for the second subplot
    axes[1].set_ylabel("Score")  # Y-axis label for the second subplot

    # Adding a title for the entire figure
    fig.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()

def plot_all_n_dim_binning_processes():

    linear_scores = lambda x: x 
    quadratic_scores = lambda x: x * x
    sqrt_scores = lambda x: sqrt(x)

    lambdas_iterable = [
        linear_scores,
        quadratic_scores,
        sqrt_scores
    ]

    plot_titles = [
        "Effect of Dimensional Averaging & Binning on Score Dist: x",
        "Effect of Dimensional Averaging & Binning on Score Dist: x^2",
        "Effect of Dimensional Averaging & Binning on Score Dist: sqrt(x)"
    ]
    
    range_lows = [
        0.3,
        0.5,
        0.0
    ]

    range_highs = [
        1.0,
        1.0,
        1.0
    ]

    for lambda_func, title, range_low, range_high in zip(lambdas_iterable, plot_titles, range_lows, range_highs):
        plot_n_dim_binning_process(dist_func = lambda_func, 
                                   dist_x_range_low = range_low, 
                                   dist_x_range_high = range_high, 
                                   plot_title = title)

# abs(ln(x)) normalization, multi-dimensional averages and binning
def n_dim_and_normalization_binning(scores_dict):

    avgs_for_all_uids = []

    for i, _ in enumerate(scores_dict['Prompt Injection']):

        avgs = []

        for analyzer1 in scores_dict:
            for analyzer2 in scores_dict:
                if analyzer1 != analyzer2:
                    avg1, avg2 = scores_dict[analyzer1][i], scores_dict[analyzer2][i]
                    avgs.append(((avg1 + avg2) / 2))

        avgs_for_all_uids.append(avgs)

    avg_for_all_uids = []

    for uid_avgs in avgs_for_all_uids:
        avg_for_all_uids.append((sum(uid_avgs) / len(uid_avgs)))

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

    scores = []

    # iterate through scores and turn each into a binned_score
    for i, score in enumerate(avg_for_all_uids):

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

        scores.append(binned_score)

    return scores, avg_for_all_uids

def plot_n_dim_and_normalization_binning_process(dist_func, dist_x_range_low, dist_x_range_high, plot_title):

    scores_dict = get_unweighted_scores_per_analyzer(dist_func, dist_x_range_low, dist_x_range_high)
    scores, unweighted_scores = n_dim_and_normalization_binning(scores_dict)
    new_unweighted_scores = sorted(unweighted_scores)
    new_scores = sorted(scores)
    uids = [x for x, _ in enumerate(scores)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # Plotting on the first subplot
    axes[0].plot(uids, new_unweighted_scores)
    axes[0].set_title("Current Score Distribution")  # Title for the first subplot
    axes[0].set_xlabel("UID")  # X-axis label for the first subplot
    axes[0].set_ylabel("Score")  # Y-axis label for the first subplot

    # Plotting on the second subplot
    axes[1].plot(uids, new_scores)
    axes[1].set_title("Normalized & Binned Score Distribution")  # Title for the second subplot
    axes[1].set_xlabel("UID")  # X-axis label for the second subplot
    axes[1].set_ylabel("Score")  # Y-axis label for the second subplot

    # Adding a title for the entire figure
    fig.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()

def plot_all_n_dim_and_normalization_binning_processes():

    linear_scores = lambda x: x 
    quadratic_scores = lambda x: x * x
    sqrt_scores = lambda x: sqrt(x)

    lambdas_iterable = [
        linear_scores,
        quadratic_scores,
        sqrt_scores
    ]

    plot_titles = [
        "Effect of Dimensional Averaging, Normalization & Binning on Score Dist: x",
        "Effect of Dimensional Averaging, Normalization & Binning on Score Dist: x^2",
        "Effect of Dimensional Averaging, Normalization & Binning on Score Dist: sqrt(x)"
    ]
    
    range_lows = [
        0.3,
        0.5,
        0.0
    ]

    range_highs = [
        1.0,
        1.0,
        1.0
    ]

    for lambda_func, title, range_low, range_high in zip(lambdas_iterable, plot_titles, range_lows, range_highs):
        plot_n_dim_and_normalization_binning_process(dist_func = lambda_func, 
                                                     dist_x_range_low = range_low, 
                                                     dist_x_range_high = range_high, 
                                                     plot_title = title)

def plot_all_processes_with_actual_dist():

    # 1. Normalization & Binning 

    unweighted_scores = get_static_miner_weights_dist() 
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
    fig.suptitle("Effect of Normalization & Binning on Score Dist: Miner Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()

    # 2. N-Dimensional Averaging

    scores_dict = {}
    for key in ['Prompt Injection', 'Sensitive Information', 'Third Analyzer', 'Fourth Analyzer']:
        scores_dict[key] = random.shuffle(get_static_miner_weights_dist().tolist(), len)

    scores, unweighted_scores = n_dim_binning(scores_dict)
    new_unweighted_scores = sorted(unweighted_scores)
    new_scores = sorted(scores)
    uids = [x for x, _ in enumerate(scores)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # Plotting on the first subplot
    axes[0].plot(uids, new_unweighted_scores)
    axes[0].set_title("Current Score Distribution")  # Title for the first subplot
    axes[0].set_xlabel("UID")  # X-axis label for the first subplot
    axes[0].set_ylabel("Score")  # Y-axis label for the first subplot

    # Plotting on the second subplot
    axes[1].plot(uids, new_scores)
    axes[1].set_title("Binned Score Distribution")  # Title for the second subplot
    axes[1].set_xlabel("UID")  # X-axis label for the second subplot
    axes[1].set_ylabel("Score")  # Y-axis label for the second subplot

    # Adding a title for the entire figure
    fig.suptitle("Effect of N-Dimensional Averaging & Binning on Score Dist: Miner Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()

    # 3. N-Dimensional Averaging, Normalization & Binning 
    scores_dict = {}
    for key in ['Prompt Injection', 'Sensitive Information', 'Third Analyzer', 'Fourth Analyzer']:
        scores_dict[key] = get_static_miner_weights_dist().tolist()

    scores, unweighted_scores = n_dim_and_normalization_binning(scores_dict)
    new_unweighted_scores = sorted(unweighted_scores)
    new_scores = sorted(scores)
    uids = [x for x, _ in enumerate(scores)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # Plotting on the first subplot
    axes[0].plot(uids, new_unweighted_scores)
    axes[0].set_title("Current Score Distribution")  # Title for the first subplot
    axes[0].set_xlabel("UID")  # X-axis label for the first subplot
    axes[0].set_ylabel("Score")  # Y-axis label for the first subplot

    # Plotting on the second subplot
    axes[1].plot(uids, new_scores)
    axes[1].set_title("Normalized & Binned Score Distribution")  # Title for the second subplot
    axes[1].set_xlabel("UID")  # X-axis label for the second subplot
    axes[1].set_ylabel("Score")  # Y-axis label for the second subplot

    # Adding a title for the entire figure
    fig.suptitle("Effect of N-Dimensional Averaging, Normalization & Binning on Score Dist: Miner Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the figure-wide title

    plt.show()


if __name__ == '__main__':

    do_all_simulations = False

    if do_all_simulations:

        plot_all_normalize_and_bin_processes()

        plot_all_n_dim_binning_processes()

        plot_all_n_dim_and_normalization_binning_processes()

    plot_all_processes_with_actual_dist()