import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Define the Data
# ---------------------------

# Class 1 data points
class1 = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [4, 5],
    [5, 5]
])

# Class 2 data points
class2 = np.array([
    [1, 0],
    [2, 1],
    [3, 1],
    [3, 2],
    [5, 3],
    [6, 5]
])

# ---------------------------
# Step 2: Compute Class Means
# ---------------------------

mu1 = np.mean(class1, axis=0)
mu2 = np.mean(class2, axis=0)

print(f"Mean of Class 1: {mu1}")
print(f"Mean of Class 2: {mu2}")

# ---------------------------
# Step 3: Compute Scatter Matrices
# ---------------------------


def compute_scatter_matrix(data, mean):
    """
    Computes the scatter matrix for a given class.

    Parameters:
        data (numpy.ndarray): Data points of the class.
        mean (numpy.ndarray): Mean vector of the class.

    Returns:
        numpy.ndarray: Scatter matrix.
    """
    scatter = np.zeros((2, 2))
    for x in data:
        x = x.reshape(2, 1)  # Convert to column vector
        mean = mean.reshape(2, 1)
        scatter += (x - mean).dot((x - mean).T)
    return scatter


# Within-class scatter matrices
S1 = compute_scatter_matrix(class1, mu1)
S2 = compute_scatter_matrix(class2, mu2)

# Total within-class scatter matrix
S_w = S1 + S2

print("\nWithin-Class Scatter Matrix S_w:")
print(S_w)

# Between-class scatter matrix
mu_diff = (mu1 - mu2).reshape(2, 1)
S_b = mu_diff.dot(mu_diff.T)

print("\nBetween-Class Scatter Matrix S_b:")
print(S_b)

# ---------------------------
# Step 4: Eigen Decomposition
# ---------------------------

# Compute the inverse of S_w
S_w_inv = np.linalg.inv(S_w)

# Compute S_w^{-1} * S_b
S_w_inv_S_b = S_w_inv.dot(S_b)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S_w_inv_S_b)

# Sort the eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Select the eigenvector with the largest eigenvalue
w = eigenvectors[:, 0]
w = w / np.linalg.norm(w)  # Normalize the eigenvector

print("\nSelected Eigenvector (w):")
print(w)

# ---------------------------
# Step 5: Project Data onto w
# ---------------------------


def project_data(data, w):
    """
    Projects data onto the vector w.

    Parameters:
        data (numpy.ndarray): Data points to project.
        w (numpy.ndarray): Projection vector.

    Returns:
        numpy.ndarray: Projected scalar values.
    """
    return data.dot(w)


# Project both classes
projected_class1 = project_data(class1, w)
projected_class2 = project_data(class2, w)

# Compute means of projections
mean_proj_class1 = np.mean(projected_class1)
mean_proj_class2 = np.mean(projected_class2)

# Classification threshold (midpoint between projected means)
threshold = (mean_proj_class1 + mean_proj_class2) / 2

print(f"\nProjected Mean of Class 1: {mean_proj_class1}")
print(f"Projected Mean of Class 2: {mean_proj_class2}")
print(f"Classification Threshold: {threshold}")

# ---------------------------
# Step 6: Visualization
# ---------------------------

plt.figure(figsize=(10, 8))

# Plot original data points
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')

# Plot class means
plt.scatter(mu1[0], mu1[1], color='darkred',
            marker='X', s=200, label='Mean Class 1')
plt.scatter(mu2[0], mu2[1], color='darkblue',
            marker='X', s=200, label='Mean Class 2')

# Plot the LDA direction
# To visualize the direction, we'll plot a line in the direction of w passing through the overall mean
overall_mean = (mu1 + mu2) / 2
line_length = 10  # Length of the line for visualization
line_points = np.array([
    overall_mean - w * line_length,
    overall_mean + w * line_length
])

plt.plot(line_points[:, 0], line_points[:, 1],
         color='green', linewidth=2, label='LDA Direction')

# Project and plot projection lines
for idx, x in enumerate(class1):
    proj = project_data(x, w) * w
    plt.plot([x[0], proj[0]], [x[1], proj[1]], 'r--', linewidth=1)

for idx, x in enumerate(class2):
    proj = project_data(x, w) * w
    plt.plot([x[0], proj[0]], [x[1], proj[1]], 'b--', linewidth=1)

# Plot the threshold on the LDA direction
threshold_point = w * threshold
plt.scatter(threshold_point[0], threshold_point[1],
            color='purple', marker='*', s=300, label='Threshold')

# Draw a perpendicular line at the threshold for classification boundary
# The normal vector to w is [-w[1], w[0]]
normal = np.array([-w[1], w[0]])
perp_line_length = 5  # Length of the perpendicular line
perp_line = np.array([
    threshold_point - normal * perp_line_length,
    threshold_point + normal * perp_line_length
])

plt.plot(perp_line[:, 0], perp_line[:, 1], color='orange',
         linestyle='--', linewidth=2, label='Classification Boundary')

# Labels and Title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Fisher Linear Discriminant Analysis (LDA) Projection')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes

plt.show()
