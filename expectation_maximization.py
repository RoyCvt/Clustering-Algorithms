import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from generator_functions import generate_gaussian_cluster, generate_random_gaussian_distribution,  plot_clusters


def expectation_step(data, weights, mean_vectors, cov_mats):
    """
    Perform the E-step of the Expectation-Maximization algorithm  
    Associates each data point with the center of its best suiting cluster  

    Parameters:
        data: The data, array of shape (num_points, num_dims)  
        weights: Mixture weights, array of shape (num_clusters,)  
        mean_vectors: Means of each cluster, array of shape (num_clusters, num_dims)  
        cov_mats: Covariance matrix of each cluster, array of shape (num_clusters, num_dims, num_dims)  

    Returns:
        membership: Probability of each data point belonging to each cluster, array of shape (num_points, num_clusters)  
        log_likelihood: Log-likelihood of the data  
    """

    num_points = data.shape[0]
    num_clusters = weights.shape[0]

    membership = np.empty((num_points, num_clusters))

    for i in range(num_clusters):
        membership[:, i] = weights[i] * multivariate_normal.pdf(data, mean_vectors[i], cov_mats[i])

    # Normalize membership grades
    membership_sum = np.sum(membership, axis=1, keepdims=True)
    membership /= membership_sum

    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(membership_sum))
    return membership, log_likelihood


def maximization_step(data, membership):
    """
    Perform the M-step of the Expectation-Maximization algorithm  
    Associates each data point with the center of its best suiting cluster  

    Parameters:
        data: The data, array of shape (num_points, num_dims)  
        membership: Probability of each data point belonging to each cluster, array of shape (num_points, num_clusters)  

    Returns:
        weights: Updated mixture weights, array of shape (num_clusters,)  
        mean_vectors: Updated means of each cluster, array of shape (num_clusters, num_dims)  
        cov_mats: Updated covariance matrix of each cluster, array of shape (num_clusters, num_dims, num_dims)  
    """
    num_points, num_dims = data.shape
    num_clusters = membership.shape[1]

    # Sums the total membership grades of each cluster for all data points
    cluster_weight_sums = np.sum(membership, axis=0)
    # Divides the weight sums by the number of points to get the weight of each cluster
    weights = cluster_weight_sums / num_points
    # Calculate a new mean vector for each cluster
    mean_vectors = np.dot(membership.T, data) / cluster_weight_sums[:, np.newaxis]
    # Create an empty array to store the covariance matrices
    cov_mats = np.empty((num_clusters, num_dims, num_dims))
    for i in range(num_clusters):
        # Calculate the difference between every point in the data and the mean vector of the current cluster
        diff = data - mean_vectors[i]
        # Calculate a new covariance matrix for the current cluster
        cov_mat = np.dot(membership[:, i] * diff.T, diff) / cluster_weight_sums[i]
        # Add the new covariance matrix to the covariance matrices array
        cov_mats[i] = cov_mat

    return weights, mean_vectors, cov_mats


def expectation_maximization(data, num_clusters, min_iters=10, max_iters=100, tol=1e-4, display_clustering=False, display_frequency=1):
    """
    Performs the Expectation-Maximization algorithm to estimate parameters for a Gaussian Mixture Model.

    Parameters:
        data: Data array of shape (num_points, num_dims)  
        num_clusters: Number of clusters in the mixture  
        min_iters: Minimum number of iterations (useful for avoiding premature convergence)  
        max_iters: Maximum number of iterations  
        tol: Tolerance for convergence  
        display_clustering: Boolean to display clustering progress  
        display_frequency: Frequency of displaying progress (every nth iteration)  

    Returns:
        weights: Mixture weights, array of shape (num_clusters,)  
        mean_vectors: Means of each cluster, array of shape (num_clusters, num_dims)  
        cov_mats: Covariance matrix of each cluster, array of shape (num_clusters, num_dims, num_dims)  
        log_likelihoods: A list of log-likelihood values at each iteration  
    """
    num_dims = data.shape[1]

    # Initialize parameters randomly
    weights = np.random.rand(num_clusters)
    weights /= np.sum(weights)
    mean_vectors = np.empty((num_clusters, num_dims))
    cov_mats = np.empty((num_clusters, num_dims, num_dims))

    for i in range(num_clusters):
        mean_vector, cov_mat = generate_random_gaussian_distribution(num_dims, 1)
        mean_vectors[i] = mean_vector
        cov_mats[i] = cov_mat

    iteration = 1
    log_likelihoods = []
    converged = False

    # Keep going while convergence wasn't achieved and the maximum number of iterations wasn't reached
    while not converged and iteration <= max_iters:
        # E-step: Compute the membership grades
        # This step calculates the probability of each data point belonging to each cluster
        membership, log_likelihood = expectation_step(data, weights, mean_vectors, cov_mats)

        # Save the log-likelihood for the current iteration
        # This helps in monitoring the convergence of the algorithm
        log_likelihoods.append(log_likelihood)

        # M-step: Update the mixture weights, mean vectors, and covariance matrices
        # This step updates the parameters based on the computed membership grades
        weights, mean_vectors, cov_mats = maximization_step(data, membership)

        if iteration >= max(2, min_iters):
            # Convergence is achieved when the change in log-likelihood is below the specified tolerance threshold
            converged = np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol

        # Display progress if requested and at the correct interval
        if display_clustering and (iteration == 1 or not iteration % display_frequency or converged):
            cluster_assignments = np.argmax(membership, axis=1)
            predicted_clusters = [data[cluster_assignments == i] for i in range(num_clusters)]
            plt.figure(figsize=(8, 6))
            plt.subplot(111, projection='3d' if num_dims == 3 else None)
            plot_clusters(predicted_clusters, mean_vectors, title=f'Expectation-Maximization: Iteration #{iteration}', show_legend=True)
            plt.show()

        iteration += 1

    if converged:
        print(f'EM converge after {iteration - 1} iterations.')
    else:
        print(f'EM did not converge after {max_iters} iterations.')

    return weights, mean_vectors, cov_mats, log_likelihoods


def main():
    num_clusters = 3
    num_dims = 3
    num_points = [600, 400, 800]
    distances = [1, 3, 7]
    np.random.seed(17845)

    # Generate random Gaussian distributions
    distributions = [generate_random_gaussian_distribution(num_dims, distances[i]) for i in range(num_clusters)]

    # Generate a random cluster from each Gaussian distribution
    clusters = [generate_gaussian_cluster(distributions[i], num_points[i], num_dims) for i in range(num_clusters)]

    # Concatenate all clusters into a single array of data points
    data = np.concatenate(clusters, axis=0)

    # Perform Expectation-Maximization
    weights, mean_vectors, cov_mats, log_likelihoods = expectation_maximization(data, num_clusters, display_clustering=True, display_frequency=10)

    # Calculate the membership grades for the final iteration
    membership, _ = expectation_step(data, weights, mean_vectors, cov_mats)

    # Determine the cluster assignment for each data point
    cluster_assignments = np.argmax(membership, axis=1)

    # Group data points into clusters based on the prediction of the algorithm
    predicted_clusters = [data[cluster_assignments == i] for i in range(num_clusters)]

    # Initialize figure with more space to fit all subplots
    plt.figure(figsize=(20, 5))

    # Plot the raw data
    plt.subplot(141, projection='3d' if num_dims == 3 else None)
    plot_clusters([data], None, 'Raw Data Scatter Plot', False)

    # Plot the predicted clusters
    plt.subplot(142, projection='3d' if num_dims == 3 else None)
    plot_clusters(predicted_clusters, mean_vectors, 'Clusters Predicted by EM', True)

    # Plot the true clusters
    plt.subplot(143, projection='3d' if num_dims == 3 else None)
    plot_clusters(clusters, None, 'True Clusters', True)

    # Plot log-likelihood over iterations
    plt.subplot(144)
    plt.plot(range(5, len(log_likelihoods)), log_likelihoods[5:])
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs. Iterations")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
