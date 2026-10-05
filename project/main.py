import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad

shell=input("Enter the shell (1s, 2s, 2p, 3s, 3p, 3d, ...): ")

a0 = 1.0 #keeping one bohr radius as a unit of length
x=1
y=1
z=1
r=(x**2 + y**2 + z**2)**0.5

# 1. Radial Probability Density Function, P(r)
def radial_probability_1s(r):
    return (4 * r**2 / a0**3) * np.exp(-2 * r / a0)

# 2. Spatial Probability Density Function, |Psi|^2
def spatial_probability_1s(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    psi = (1.0 / np.sqrt(np.pi * a0**3)) * np.exp(-r / a0)
    return psi**2

def spatial_probability_2s(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    psi = (
        1.0 / np.sqrt(32 * np.pi * a0**3)
        * (2 - r / a0)
        * np.exp(-r / (2 * a0))
    )

    return psi**2

def plot_orbital(probability, X, Y, Z):
    threshold = 0.01 * probability.max()
    mask = probability >= threshold

    fig = go.Figure(
        data=go.Scatter3d(
            x=X[mask],
            y=Y[mask],
            z=Z[mask],
            mode="markers",
            marker=dict(
                size=2,
                color=probability[mask],
                opacity=0.15
            )
        )
    )

    fig.update_layout(
        title="Hydrogen Orbital",
        scene=dict(
            xaxis_title="x (a₀)",
            yaxis_title="y (a₀)",
            zaxis_title="z (a₀)",
            aspectmode="cube"
        )
    )

    fig.show()

if shell == "1s" or shell == "1S":
    print("Radial Probability Density at r={}:".format(r), radial_probability_1s(r))
    print("Spatial Probability Density at (x=1, y=1, z=1):", spatial_probability_1s(x, y, z))

    coordinates = np.linspace(-3 * a0, 3 * a0, 60)
    X, Y, Z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    density = spatial_probability_1s(X, Y, Z)

    plot_orbital(density, X, Y, Z)

    total_probability, error = quad(
    radial_probability_1s,
    0,
    np.inf
)#checking the normalization of the wavefunction by integrating the radial probability density over all space
    print("\n--- Normalization Check ---")
    print("Total probability:", total_probability)
    print("Integration error:", error)

    if abs(total_probability - 1) < 1e-10:
        print("PASS: Wavefunction is normalized.")
    else:
        print("FAIL: Wavefunction is not normalized.")
