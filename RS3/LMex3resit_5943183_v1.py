import numpy as np
import matplotlib.pyplot as plt

'''
Write your own Python/Matlab implementation of this problem, where the number of elements
can be varied. No input is provided, the code can be specifically written for the current problem
(i.e., boundary conditions, length, load etc. can all be hardcoded). You have to upload the code
(as .py or .m file), which upon running creates the plot(s) or the data used to solve question e.
(lay-out of the plots may be different). Use your expression for the force vector obtained in b.
If you struggle with coding, then please solve the problem using 1 and 3 elements (either hardcoded or by hand), 
this leads to a maximum of 36/40 for this assignment 
'''

# Hardcoded beam properties
L = 750e-3  # Total length of the beam (m) (750 mm)
h = 35e-3  # Height of the beam section (m) (35mm)
w = 15e-3  # Thickness of the beam section (m) (15mm)
E = 210e9  # Young's modulus (Pa) (210 GPa)
a = 0.02e6  # Linear variation coefficient for q_y (N/m^2) (0.02 N/mm^2)
b = 3e3  # Constant term for q_y (N/m) (3 N/mm)
F_y = 0.5e3  # Point load (N, upwards)
n_elements = 10  # Number of finite elements

# Moment of inertia of the beam cross-section
I = (w * h**3) / 12  # Moment of inertia (m^4)

# Derived properties
n_nodes = n_elements + 1  # Total number of nodes

# Generate node positions and element lengths
x_nodes = np.linspace(0, L, n_nodes)
element_length = L / n_elements

# Shape function derivatives for a beam element (in local coordinates)
def beam_shape_func_derivatives(xi):
    return np.array([
        -3 + 4 * xi - xi**2,
        -1 + xi - xi**2,
        3 - 4 * xi + xi**2,
        -xi + xi**2,
    ])

# Generate global stiffness matrix and force vector
global_K = np.zeros((2 * n_nodes, 2 * n_nodes))
global_F = np.zeros(2 * n_nodes)

for e in range(n_elements):
    # Element start and end nodes
    n1, n2 = e, e + 1

    # Element stiffness matrix (local)
    l = element_length
    ke = E * I / l**3 * np.array([
        [12, 6 * l, -12, 6 * l],
        [6 * l, 4 * l**2, -6 * l, 2 * l**2],
        [-12, -6 * l, 12, -6 * l],
        [6 * l, 2 * l**2, -6 * l, 4 * l**2],
    ])

    # Element force vector due to q_y = ax + b
    xe1, xe2 = x_nodes[n1], x_nodes[n2]
    q1, q2 = a * xe1 + b, a * xe2 + b
    fe = l / 20 * np.array([
        7 * q1 + 3 * q2,
        l * (3 * q1 + 2 * q2),
        3 * q1 + 7 * q2,
        -l * (2 * q1 + 3 * q2),
    ])

    # Assemble into global matrix and vector
    global_dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
    for i in range(4):
        global_F[global_dofs[i]] += fe[i]
        for j in range(4):
            global_K[global_dofs[i], global_dofs[j]] += ke[i, j]

# Apply boundary conditions (cantilever at x = 0)
global_K[0, :] = 0
global_K[:, 0] = 0
global_K[1, :] = 0
global_K[:, 1] = 0
global_K[0, 0] = 1
global_K[1, 1] = 1

global_F[0] = 0
global_F[1] = 0
global_F[-2] -= F_y  # Vertical displacement DOF of the last node

# Solve for displacements
displacements = np.linalg.solve(global_K, global_F)

# Extract displacements and rotations
v = displacements[0::2]  # Transverse displacements
phi = displacements[1::2]  # Rotations

# Calculate bending moments and shear forces
bending_moments = np.zeros(n_elements + 1)
shear_forces = np.zeros(n_elements + 1)

for e in range(n_elements):
    n1, n2 = e, e + 1
    l = element_length
    ve = np.array([v[n1], phi[n1], v[n2], phi[n2]])
    
    # Element bending moment and shear force
    bending_moments[n1] = -(E * I / l**2) * np.dot([-6 / l, -4, 6 / l, -2], ve)
    shear_forces[n1] = -(E * I / l**3) * np.dot([-12, -6 * l, 12, -6 * l], ve)

# Add final bending moment and shear force
bending_moments[-1] = bending_moments[-2]
shear_forces[-1] = shear_forces[-2]

# Plot results
plt.figure(figsize=(12, 8))

# Plot transverse displacement
plt.subplot(2, 2, 1)
plt.plot(x_nodes, v, marker="o", label="Displacement v(x)")
plt.xlabel("x (m)")
plt.ylabel("Displacement (m)")
plt.title("Transverse Displacement")
plt.legend()
plt.grid()

# Plot rotation
plt.subplot(2, 2, 2)
plt.plot(x_nodes, phi, marker="o", label="Rotation \u03C6(x)")
plt.xlabel("x (m)")
plt.ylabel("Rotation (rad)")
plt.title("Rotation")
plt.legend()
plt.grid()

# Plot bending moment
plt.subplot(2, 2, 3)
plt.plot(x_nodes, bending_moments, marker="o", label="Bending Moment M(x)")
plt.xlabel("x (m)")
plt.ylabel("Bending Moment (Nm)")
plt.title("Bending Moment")
plt.legend()
plt.grid()

# Plot shear force
plt.subplot(2, 2, 4)
plt.plot(x_nodes, shear_forces, marker="o", label="Shear Force Q(x)")
plt.xlabel("x (m)")
plt.ylabel("Shear Force (N)")
plt.title("Shear Force")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
