import numpy as np
import matplotlib.pyplot as plt
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Function to compute the stiffness matrix for one element
def stiffness_matrix_element(elastic_module: float, surface: float, l: float):
    A = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]])  # Stiffness matrix for quadratic elements
    # Return the element stiffness matrix scaled by material properties and element length
    return elastic_module * surface / (3 * l) * A

# Function to assemble the global stiffness matrix from the element stiffness matrices
def assemble_global_stiffness_matrix(nelem: int, ltot: float, elastic_module: float, surface: float):
    n = 3 + (nelem - 1) * 2  # Total number of nodes (3 nodes per element with shared ones)
    l = ltot / nelem  # Length of each element
    K_global = np.zeros((n, n))  # Initialize global stiffness matrix
    for elem in range(nelem):  # Loop over all elements
        # Add element stiffness matrix to the global matrix at the correct positions
        K_global[2 * elem:2 * elem + 3, 2 * elem:2 * elem + 3] += stiffness_matrix_element(elastic_module, surface, l)
    return K_global  # Return the assembled global stiffness matrix

# Function to create a simple load vector using a linear and quadratic load distribution
def load_vector_simple(coeff_a: float, coeff_b: float, nelem: int, ltot: float):
    n = 3 + (nelem - 1) * 2  # Total number of nodes
    nodes = np.linspace(0, ltot, n)  # Positions of the nodes
    load = np.zeros(n)  # Initialize load vector
    dummy = coeff_a * nodes + coeff_b / 2 * nodes ** 2  # Compute the load based on a linear-quadratic function
    load[1:] = np.diff(dummy)  # Calculate the difference between consecutive nodes to define the load
    return load  # Return the load vector

# Helper functions for shape functions used in load vector calculation
def f1(A, B, C, D, x, l_el):
    return 2/l_el**2 * (A*B*C*x + 0.5*A*B*D*x**2 - 0.5*A*C*x**2 - 1/3*A*D*x**3 - 0.5*B*C*x**2 - 1/3*B*D*x**3 + 1/3*C*x**3 + 1/4*D*x**4)

def f2(A, B, C, D, x, l_el):
    return -4/l_el**2 * (A*B*C*x + 0.5*A*B*D*x**2 - 0.5*A*C*x**2 - 1/3*A*D*x**3 - 0.5*B*C*x**2 - 1/3*B*D*x**3 + 1/3*C*x**3 + 1/4*D*x**4)

def f3(A, B, C, D, x, l_el):
    return 2/l_el**2 * (A*B*C*x + 0.5*A*B*D*x**2 - 0.5*A*C*x**2 - 1/3*A*D*x**3 - 0.5*B*C*x**2 - 1/3*B*D*x**3 + 1/3*C*x**3 + 1/4*D*x**4)

# Function to compute load vector using shape functions
def load_vector_shape_functions(coeff_a: float, coeff_b: float, nelem: int, ltot: float):
    n = 3 + (nelem - 1) * 2  # Total number of nodes
    l = ltot / nelem  # Length of each element
    load = np.zeros(n)  # Initialize load vector
    nodes = np.linspace(0, ltot, n)  # Node positions
    for ii in range(nelem):  # Loop over elements
        x_i = nodes[2 * ii]  # Position of first node of the element
        x_j = nodes[2 * ii + 1]  # Position of middle node
        x_k = nodes[2 * ii + 2]  # Position of last node
        # Use shape functions to compute the load contribution of each node in the element
        load[2 * ii] += f1(x_j, x_k, coeff_a, coeff_b, x_k, l) - f1(x_j, x_k, coeff_a, coeff_b, x_i, l)
        load[2 * ii + 1] += f2(x_i, x_k, coeff_a, coeff_b, x_k, l) - f2(x_i, x_k, coeff_a, coeff_b, x_i, l)
        load[2 * ii + 2] += f3(x_i, x_j, coeff_a, coeff_b, x_k, l) - f3(x_i, x_j, coeff_a, coeff_b, x_i, l)
    return load  # Return the computed load vector

# Function to apply problem-specific boundary conditions (remove first row and column)
def apply_problem_speficic_bc(K_global, load):
    K_global = K_global[1:, 1:]  # Remove the first row and column of the stiffness matrix
    load = load[1:]  # Remove the first value from the load vector
    return K_global, load  # Return modified stiffness matrix and load vector

# Function to solve the problem for a given number of elements, material properties, and load coefficients
def quadratic_elements_problem(ltot, surface, elastic_module, coeff_a, coeff_b, nelem):
    F = load_vector_shape_functions(coeff_a, coeff_b, nelem, ltot)  # Compute the load vector
    K = assemble_global_stiffness_matrix(nelem, ltot, elastic_module, surface)  # Assemble global stiffness matrix
    K_reduced, F_reduced = apply_problem_speficic_bc(K, F)  # Apply boundary conditions
    u_partial = np.linalg.solve(K_reduced, F_reduced)  # Solve the reduced system for displacements
    u_complete = np.zeros(3 + (nelem - 1) * 2)  # Initialize the full displacement vector
    u_complete[1:] = u_partial  # Add the computed displacements (except the boundary condition point)
    return u_complete  # Return the computed displacement vector

# Function to compute true displacement, error, and interpolated solution for comparison
def true_displacement_and_error(coeff_a, coeff_b, surface, elastic_module, sol, ltot, nelem, number_points):
    n = 3 + (nelem - 1) * 2  # Total number of nodes
    nodes = np.linspace(0, ltot, n)  # Node positions
    l = ltot/nelem  # Length of each element
    nodes_dense = np.linspace(0, ltot, number_points * nelem - (nelem - 1))  # Dense grid of points for interpolation
    sol_dense = np.zeros_like(nodes_dense)  # Initialize dense solution array
    for ii in range(nelem):  # Loop over each element
        x_vals = nodes[2 * ii:2 * ii + 3]  # Extract node positions for the current element
        y_vals = sol[2 * ii:2 * ii + 3]  # Extract displacements at these nodes

        A = np.array([  # Setup system to find quadratic polynomial coefficients
            [x_vals[0] ** 2, x_vals[0], 1],
            [x_vals[1] ** 2, x_vals[1], 1],
            [x_vals[2] ** 2, x_vals[2], 1]
        ])
        coeffs = np.linalg.solve(A, y_vals)  # Solve for coefficients of quadratic polynomial
        a_, b_, c_ = coeffs

        # Define quadratic polynomial using the computed coefficients
        def quadratic_polynomial(x):
            return a_ * x ** 2 + b_ * x + c_

        # Interpolate solution over the dense grid for the current element
        sol_dense[(number_points - 1) * ii:(number_points - 1) * ii + number_points] = quadratic_polynomial(
            nodes_dense[(number_points - 1) * ii:(number_points - 1) * ii + (number_points)])

    # Compute the true displacement and the RMS error between true and computed solution
    true = 1/(surface * elastic_module) * (-(coeff_a / 2 * nodes_dense**2 + coeff_b / 6 * nodes_dense**3) + nodes_dense * (coeff_a * ltot + coeff_b * ltot**2 / 2))
    error_ = np.sqrt(np.sum((true - sol_dense) ** 2) / len(true))

    return true, error_, sol_dense, nodes_dense  # Return true solution, error, interpolated solution, and dense nodes

# INPUT VALUES
L = 500  # Total length
Area = 120  # Cross-sectional area
E = 70e3  # Elastic modulus
a = 13.  # Load coefficient (linear term)
b = 0.13  # Load coefficient (quadratic term)
N = 5  # Number of elements

# Solve for displacements and compute the true solution
u = quadratic_elements_problem(L, Area, E, a, b, N)
u_true, err, u_dense, n_dense = true_displacement_and_error(a, b, Area, E, u, L, N, 500)

# Plot computed solution and true solution
plt.figure(figsize=(6, 5))
plt.plot(n_dense, u_dense, label='Computed')
plt.plot(n_dense, u_true, 'y-.', label='Analytical')
plt.legend()
plt.grid()
plt.xlabel('# elements')
plt.ylabel('Displacement of last node [mm]')
# Save the figure in the same folder as the script
    # Define the file name for the figure
fig_name = 'displacement_delta.png'
plt.savefig(os.path.join(script_dir, fig_name), dpi=300)

# CONVERGENCE PROBLEM
nmax = 10  # Maximum number of elements to test for convergence
u_end = np.zeros(nmax)  # Array to store displacements of last node for each element count
err = np.zeros(nmax)  # Array to store RMS error for each element count
for i in range(1, nmax + 1):
    u = quadratic_elements_problem(L, Area, E, a, b, i)  # Solve for displacement with 'i' elements
    u_end[i - 1] = u[-1]  # Store displacement of last node
    u_true, error, u_dense, n_dense = true_displacement_and_error(a, b, Area, E, u, L, i, 500)  # Compute true solution and error
    err[i - 1] = error  # Store the RMS error

# Plot convergence study (error vs number of elements)
plt.figure(figsize=(6, 5))
plt.plot(np.arange(1, nmax + 1), err)
plt.grid()
plt.xlabel('# elements')
plt.ylabel('RMS delta analytical-computed solution [mm]')
# Save the figure in the same folder as the script
    # Define the file name for the figure
fig_name = 'delta_convergence.png'
plt.savefig(os.path.join(script_dir, fig_name), dpi=300)