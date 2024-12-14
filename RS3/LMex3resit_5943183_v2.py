import numpy as np
import matplotlib.pyplot as plt

# Material and geometry properties
E = 210e9  # Young's modulus in Pa
L = 750e-3  # Length of the beam in m
h = 35e-3  # Height of the beam in m
w = 15e-3  # Thickness of the beam in m
I = (w * h**3) / 12  # Second moment of area (rectangular cross-section)

# Loading parameters
a = 0.02e6  # Distributed load coefficient (N/m^2)
b = 3.0e3  # Distributed load intercept (N/m)
P = 500.0  # Point load in N

# Discretization parameters
num_elements = 50  # Number of elements along the beam
num_nodes = 2 * num_elements + 2  # Nodes per element include displacement and rotation

dx = L / num_elements  # Length of each element

# Assemble load vector
x_coords = np.linspace(0, L, num_elements + 1)
q_loads = (a * x_coords + b)  # Linearly varying distributed load

f = np.zeros(num_nodes)  # Initialize force vector
for i in range(num_elements):
    x1 = x_coords[i]
    x2 = x_coords[i + 1]
    q1 = a * x1 + b
    q2 = a * x2 + b
    avg_q = (q1 + q2) / 2
    f[2 * i] += avg_q * dx / 2
    f[2 * i + 2] += avg_q * dx / 2

# Add point load at the end
f[-2] -= P

# Assemble stiffness matrix
K = np.zeros((num_nodes, num_nodes))
for i in range(num_elements):
    k_local = (E * I / dx**3) * np.array([
        [12, 6 * dx, -12, 6 * dx],
        [6 * dx, 4 * dx**2, -6 * dx, 2 * dx**2],
        [-12, -6 * dx, 12, -6 * dx],
        [6 * dx, 2 * dx**2, -6 * dx, 4 * dx**2]
    ])

    dof = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
    for ii in range(4):
        for jj in range(4):
            K[dof[ii], dof[jj]] += k_local[ii, jj]

# Apply boundary conditions (fixed at x = 0)
K[0, :] = 0
K[1, :] = 0
K[:, 0] = 0
K[:, 1] = 0
K[0, 0] = 1
K[1, 1] = 1

f[0] = 0
f[1] = 0

# Solve for displacements
d = np.linalg.solve(K, f)

# Extract deflection and rotation
deflection = d[0::2]
rotation = d[1::2]

# Compute bending moment and tensile stress
bending_moment = -E * I * np.gradient(rotation, dx)
tensile_stress = bending_moment * (h / 2) / I

# Adjust x-coordinates for plotting deflection and rotation
x_coords_mid = np.linspace(0, L, len(deflection))

# Plot results
plt.figure(figsize=(12, 8))

# Plot deflection
plt.subplot(3, 1, 1)
plt.plot(x_coords_mid, deflection, label="Deflection (y-direction)")
plt.xlabel("x (m)")
plt.ylabel("Deflection (m)")
plt.title("Deflection of the Beam")
plt.legend()
plt.grid()

# Plot bending moment
plt.subplot(3, 1, 2)
plt.plot(x_coords_mid, bending_moment, label="Bending Moment", color="orange")
plt.xlabel("x (m)")
plt.ylabel("Bending Moment (Nm)")
plt.title("Bending Moment Distribution")
plt.legend()
plt.grid()

# Plot tensile stress
plt.subplot(3, 1, 3)
plt.plot(x_coords_mid, tensile_stress, label="Tensile Stress", color="green")
plt.xlabel("x (m)")
plt.ylabel("Stress (Pa)")
plt.title("Tensile Stress Distribution")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()