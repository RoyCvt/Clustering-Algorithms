from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


def generate_random_mean_vector(num_dims: int, distance: float) -> np.ndarray:
    """
    Generates a random mean vector of size num_dims
    The distance parameter specifies the distance of the mean from the origin
    """
    return np.random.rand(num_dims) * distance


def generate_random_covariance_matrix(num_dims: int, max_attempts: int = 100, tol: float = 1e-3) -> np.ndarray:
    """
    Generates a random, valid covariance matrix of size num_dims x num_dims
    """
    for _ in range(max_attempts):
        # Create a matrix of size num_dims x num_dims with values between -0.5 and 0.5 (centered around 0)
        cov_mat = np.random.rand(num_dims, num_dims) - 0.5
        # Multiply the matrix by its transpose to make it symmetric (and valid)
        cov_mat = np.dot(cov_mat, cov_mat.T)

        # Check if all eigenvalues are positive
        if np.all(np.linalg.eigvals(cov_mat) > tol):
            return cov_mat

    # If unable to find a valid matrix after max_attempts
    raise ValueError(f"Unable to generate a positive definite covariance matrix after {max_attempts} attempts.")


def generate_random_gaussian_distribution(num_dims: int, distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random gaussian distribution with mean "mean_vector" and covariance matrix "cov_mat"
    The distance parameter specifies the distance of the mean from the origin
    """
    # Create a random mean vector
    mean_vector = generate_random_mean_vector(num_dims, distance)
    # Create a random covariance matrix
    cov_mat = generate_random_covariance_matrix(num_dims)
    return mean_vector, cov_mat


def generate_normal_data_point(gaussian_dist: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Generate a single data point from a gaussian distribution (mean vector and covariance matrix) gaussian_dist.
    """
    return np.random.multivariate_normal(*gaussian_dist, check_valid="warn", tol=1e-8)


def generate_gaussian_cluster(gaussian_dist: Tuple[np.ndarray, np.ndarray], num_points: int, num_dims: int) -> np.ndarray:
    """
    Generate a cluster of num_points data points from the gaussian distribution gaussian_dist.
    gaussian_dist is a tuple of the form (mean vector, covariance matrix).
    num_dims is the number of dimensions in the distribution.
    """
    # Create an empty array to store the points that form the cluster
    cluster = np.empty((num_points, num_dims))
    for i in range(num_points):
        # Generate a random data point
        data_point = generate_normal_data_point(gaussian_dist)
        # Add the new data point to the cluster
        cluster[i] = data_point
    return cluster


def generate_random_gaussian_clusters(num_clusters: int, num_points: int, num_dims: int, distance_range: Tuple[float, float]) -> list[np.ndarray]:
    """
    Generates random clusters of points from random gaussian distributions.
    num_clusters is the amount of gaussian clusters generated.
    num_points is the amount of data points in each cluster.
    num_dims is the number of dimensions in the distribution.
    distance_range is a tuple describing the range for the distances of the distributions.
    """
    # Create an empty list to store the clusters
    cluster_list = []
    for _ in range(num_clusters):
        # Randomize a "distance" in the specified range
        distance = round(np.random.uniform(*distance_range), ndigits=3)
        # Generate a random gaussian distribution
        gaussian_dist = generate_random_gaussian_distribution(num_dims, distance)
        # Generate a cluster of data points from the gaussian distribution
        cluster = generate_gaussian_cluster(gaussian_dist, num_points, num_dims)
        # Add the new cluster to the cluster list
        cluster_list.append(cluster)
    return cluster_list


def generate_random_cluster_centroids(num_clusters: int, num_dims: int) -> list[np.ndarray]:
    """
    Generates random initial points for cluster centroids.
    num_clusters is the amount of centroids generated (1 centroid generated per cluster).
    num_dims is the number of dimensions of the generated centroids.
    """
    # Create an empty list to store the centroids
    cluster_centroid_list = []
    for _ in range(num_clusters):
        # Generate a random distance for the centroid
        distance = np.random.rand()
        # Generate a random mean vector to serve as the centroid of a cluster
        cluster_centroid = generate_random_mean_vector(num_dims, distance)
        # Add the new centroid to the centroid list
        cluster_centroid_list.append(cluster_centroid)
    return cluster_centroid_list


def generate_line_shaped_gaussian_distribution(start_point: np.ndarray, end_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the mean vector and covariance matrix for a Gaussian distribution that follows
    a line segment defined by two points in a multi-dimensional space.

    Parameters:
        start_point (np.ndarray): The starting point of the line segment, array of shape (num_dims,).  
        end_point (np.ndarray): The ending point of the line segment, array of shape (num_dims,).  

    Returns:
        mean_vector (np.ndarray): Means of the distribution, array of shape (num_dims,)  
        cov_mat (np.ndarray): Covariance matrix of the distribution, array of shape (num_dims, num_dims)  
    """
    # Calculate the vector from start_point to end_point
    vector = end_point - start_point
    # Calculate the length of the vector
    vector_norm = np.linalg.norm(vector)
    # Normalize the vector to have a length of 1
    unit_vector = vector / vector_norm
    # Create the covariance matrix based on the vector length
    cov_mat = vector_norm * np.outer(unit_vector, unit_vector)
    # Add a small epsilon to the covariance matrix to avoid singularity
    epsilon = 0.005 * np.identity(cov_mat.shape[0])
    cov_mat += epsilon
    # Calculate the mean vector as the midpoint of the line segment
    mean_vector = start_point + (end_point - start_point) / 2
    return mean_vector, cov_mat


def generate_polygon_of_clusters(polygon_vertices: List[np.ndarray], num_points_per_cluster: List[int]) -> List[np.ndarray]:
    """
    Generates clusters of data points arranged in a polygonal shape, where each cluster is
    defined by a line segment between consecutive points in the vertices list.

    Parameters:
        polygon_vertices (List[np.ndarray]): A list of points that define the vertices of the polygon, 
                                             each point in the list is an array of shape (num_dims,).  
        num_points_per_cluster (List[int]): A list containing the number of points to generate for each cluster.  
        
    Returns:
        cluster_list (List[np.ndarray]): A list containing clusters of points.  
                                         Each cluster is a NumPy array of shape (num_points, num_dims).
    """
    num_dims = len(polygon_vertices[0])

    # Create an empty list to store the clusters
    cluster_list = []
    # Generate a cluster for each pair of consecutive polygon vertices
    for cur_index, cur_vertex in enumerate(polygon_vertices):
        # Get the previous vertex of the polygon
        prev_vertex = polygon_vertices[cur_index - 1]
        # Generate a line-shaped gaussian distribution between the previous vertex and the current one
        gaussian_dist = generate_line_shaped_gaussian_distribution(prev_vertex, cur_vertex)
        # Generate a cluster of data points from the gaussian distribution
        cluster = generate_gaussian_cluster(gaussian_dist, num_points_per_cluster[cur_index], num_dims)
        # Add the new cluster to the cluster list
        cluster_list.append(cluster)
    return cluster_list


def plot_clusters(clusters: list[np.ndarray], mean_vectors: np.ndarray = None, title: str = None, show_legend: bool = True) -> None:
    """
    Plots clusters of points in 2D or 3D space depending on the number of dimensions.

    Parameters:
        clusters (list of numpy.ndarray): A list of NumPy arrays, where each array has the shape (num_points, num_dims).
                                          All clusters must have points with the same number of dimensions, either 2D or 3D.
        mean_vectors (numpy.ndarray, optional): A NumPy array of shape (num_clusters, num_dims), where each row represents the mean vector
                                                of the Gaussian distribution that generated the corresponding cluster.
        title (str, optional): A string representing the title of the plot.
                               If not provided, the default title will be used based on the number of dimensions
                               (e.g., '2D Cluster Plot' or '3D Cluster Plot').
        show_legend (bool, optional): A boolean flag indicating whether to display a legend on the plot.
                                      Defaults to True. The legend shows labels for each cluster and, if provided,
                                      their corresponding mean vectors.

    Raises:
        ValueError: If the clusters have inconsistent dimensionality.
        ValueError: If the number of dimensions is smaller than 2.
    """
    num_dims = clusters[0].shape[1]

    if not all(cluster.shape[1] == num_dims for cluster in clusters):
        raise ValueError("All clusters must have the same number of dimensions.")

    ax = plt.gca()

    if num_dims in (2, 3):
        for i, cluster in enumerate(clusters):
            ax.scatter(*[cluster[:, dim] for dim in range(num_dims)], alpha=0.5, label=f"Cluster {i+1}")
            if mean_vectors is not None:
                # Mark the mean of the cluster using its number
                ax.text(*[mean_vectors[i][dim] for dim in range(num_dims)], str(i + 1), fontsize=12, fontweight="bold", color="black", ha="center", va="center")

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        if num_dims == 3:
            ax.set_zlabel("Z-axis")

        ax.set_title(title if title else f"{num_dims}D Cluster Plot")
        if show_legend:
            ax.legend()
    else:
        raise ValueError(f"The number of dimensions of the points in the clusters should be 2 or 3, but {num_dims} was found.")


def main():
    num_clusters = 3
    num_points = 1000
    num_dims = 3
    distance_range = (1, 10)
    rows = 2
    cols = 3

    plt.figure(figsize=(cols * 5, rows * 5))

    for row in range(rows):
        for col in range(cols):
            clusters = generate_random_gaussian_clusters(num_clusters, num_points, num_dims, distance_range)
            plt.subplot(rows, cols, cols * row + col + 1, projection="3d" if num_dims == 3 else None)
            plot_clusters(clusters, mean_vectors=None, title=None, show_legend=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
