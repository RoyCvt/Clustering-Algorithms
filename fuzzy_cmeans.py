import matplotlib.pyplot as plt
import numpy as np

from generator_functions import (
    generate_gaussian_cluster,
    generate_random_cluster_centroids,
    generate_random_gaussian_distribution,
    plot_clusters,
)


def fuzzy_cmeans(data, num_clusters, fuzziness, min_iters=10, max_iters=100, tol=1e-4, display_clustering=False, display_frequency=1):
    """
    Performs the Fuzzy C-Means algorithm to estimate parameters for a Gaussian Mixture Model.

    Parameters:
        data: Data array of shape (num_points, num_dims).
        num_clusters: Number of clusters in the mixture.
        fuzziness: Degree of overlap between clusters.
                   When fuzziness is 1 the algorithm effectively becomes Hard C-Means.
                   Larger value can be useful when dealing with overlapping or ambiguous data, while a smaller value might be suitable for achieving more distinct clusters.
        min_iters: Minimum number of iterations (useful for avoiding premature convergence).
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence.
        display_clustering: Boolean to display clustering progress.
        display_frequency: Frequency of displaying progress (every nth iteration).

    Returns:
        centroids: Cluster centroids array of shape (num_clusters, num_dims).
        membership: Membership array of shape (num_points, num_clusters).
        iteration_loss: A list of loss values at each iteration.
    """
    num_dims = data.shape[1]

    # Initialize cluster centroids randomly
    centroids = generate_random_cluster_centroids(num_clusters, num_dims)

    converged = False
    iteration_loss = []
    prev_centroids = None
    prev_membership = None
    iteration = 1

    # Keep going while convergence wasn't achieved and the maximum number of iterations wasn't reached
    while not converged and iteration <= max_iters:
        # Calculate the Euclidean distance between each data point to each cluster centroid
        data_distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        # Calculating membership values
        membership = 1 / (data_distances ** (2 / (fuzziness - 1)))
        # Normalizing membership values
        membership = membership / np.sum(membership, axis=1)[:, None]

        # Update cluster centroids
        fuzzy_membership = membership**fuzziness
        centroids = np.dot(fuzzy_membership.T, data) / np.sum(fuzzy_membership, axis=0)[:, None]

        # Calculate the loss function
        loss = np.sum((data_distances**2) * fuzzy_membership)
        iteration_loss.append(loss)

        if iteration >= max(2, min_iters):
            # Loss convergence is achieved when the change in the loss between consecutive iterations is below the specified tolerance threshold
            loss_difference = np.abs(iteration_loss[-1] - iteration_loss[-2])
            loss_converged = loss_difference < tol

            # Centroids convergence is achieved when the change in the centroids position between consecutive iterations is below the specified tolerance threshold
            centroid_shift = np.max(np.linalg.norm(centroids - prev_centroids, axis=1))
            centroids_converged = centroid_shift < tol

            # Membership convergence is achieved when the change in the membership values between consecutive iterations is below the specified tolerance threshold
            membership_shift = np.max(np.abs(membership - prev_membership))
            membership_converged = membership_shift < tol

            # Convergence occurs when all the different ___ converge
            converged = loss_converged and centroids_converged and membership_converged

        # Update previous values
        prev_centroids = centroids.copy()
        prev_membership = membership.copy()

        # Display progress if requested and at the correct interval
        if display_clustering and (iteration == 1 or not iteration % display_frequency or converged):
            cluster_assignments = np.argmax(membership, axis=1)
            predicted_clusters = [data[cluster_assignments == i] for i in range(num_clusters)]
            plt.figure(figsize=(8, 6))
            plt.subplot(111, projection="3d" if num_dims == 3 else None)
            plot_clusters(predicted_clusters, centroids, title=f"Fuzzy C-Means: Iteration #{iteration}", show_legend=True)
            plt.show()

        iteration += 1

    if converged:
        print(f"FCM converge after {iteration - 1} iterations.")
    else:
        print(f"FCM did not converge after {max_iters} iterations.")

    return centroids, membership, iteration_loss


def main():
    num_clusters = 3
    fuzziness = 2
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

    # Perform Fuzzy C-Means
    centroids, membership, iteration_loss = fuzzy_cmeans(data, num_clusters, fuzziness, display_clustering=True, display_frequency=10)

    # Determine the cluster assignment for each data point
    cluster_assignments = np.argmax(membership, axis=1)

    # Group data points into clusters based on the prediction of the algorithm
    predicted_clusters = [data[cluster_assignments == i] for i in range(num_clusters)]

    # Initialize figure with more space to fit all subplots
    plt.figure(figsize=(20, 5))

    # Plot the raw data
    plt.subplot(141, projection="3d" if num_dims == 3 else None)
    plot_clusters([data], None, "Raw Data Scatter Plot", False)

    # Plot the predicted clusters
    plt.subplot(142, projection="3d" if num_dims == 3 else None)
    plot_clusters(predicted_clusters, centroids, "Clusters Predicted by FCM", True)

    # Plot the true clusters
    plt.subplot(143, projection="3d" if num_dims == 3 else None)
    plot_clusters(clusters, None, "True Clusters", True)

    # Plot log-likelihood over iterations
    plt.subplot(144)
    plt.plot(range(5, len(iteration_loss)), iteration_loss[5:])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iterations")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
