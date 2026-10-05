import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

if shell == "1s" or shell == "1S":
    print("Radial Probability Density at r={}:".format(r), radial_probability_1s(r))
    print("Spatial Probability Density at (x=1, y=1, z=1):", spatial_probability_1s(x, y, z))

    fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{}, {'type': 'scene'}]],
    subplot_titles=(
        'Radial Probability Density ($P(r)$)',
        '3D 1s Spatial Probability Density',
    ),
)

# Left Plot: Radial Probability Density
    r_vals = np.linspace(0, 5 * a0, 500)
    P_vals = radial_probability_1s(r_vals)

    fig.add_trace(
    go.Scatter(
        x=r_vals,
        y=P_vals,
        mode='lines',
        line=dict(color='royalblue', width=2),
        name='$4\\pi r^2 |\\psi_{1s}|^2$',
    ),
    row=1,
    col=1,
)
    fig.add_trace(
    go.Scatter(
        x=[a0, a0],
        y=[0, float(np.max(P_vals))],
        mode='lines',
        line=dict(color='crimson', dash='dash'),
        name='Bohr Radius ($a_0$)',
    ),
    row=1,
    col=1,
)
    fig.update_xaxes(title_text='Distance from nucleus $r$ ($a_0$)', row=1, col=1)
    fig.update_yaxes(title_text='Probability Density', row=1, col=1)

# Right Plot: 3D Electron Cloud
    coordinates = np.linspace(-3 * a0, 3 * a0, 60)
    X, Y, Z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    density = spatial_probability_1s(X, Y, Z)

    fig.add_trace(
    go.Isosurface(
        x=X.ravel(),
        y=Y.ravel(),
        z=Z.ravel(),
        value=density.ravel(),
        isomin=float(np.max(density) * 0.03),
        isomax=float(np.max(density) * 0.8),
        surface_count=6,
        opacity=0.3,
        colorscale='Magma',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title='Probability Density $|\\psi|^2$', x=1.02, y=0.5, len=0.75),
        name='Electron cloud density',
    ),
    row=1,
    col=2,
)
    fig.update_layout(
    title_text='1s Hydrogen Orbital Probability Densities',
    width=1300,
    height=700,
    template='plotly_white',
    margin=dict(r=150),
    legend=dict(x=0.03, y=0.97, xanchor='left', yanchor='top'),
    scene=dict(
        xaxis_title='x ($a_0$)',
        yaxis_title='y ($a_0$)',
        zaxis_title='z ($a_0$)',
        aspectmode='cube',
    ),
)
    fig.show()

    total_probability, error = quad(
    radial_probability_1s,
    0,
    np.inf
)#cheking the normalization of the wavefunction by integrating the radial probability density over all space
    print("\n--- Normalization Check ---")
    print("Total probability:", total_probability)
    print("Integration error:", error)

    if abs(total_probability - 1) < 1e-10:
        print("PASS: Wavefunction is normalized.")
    else:
        print("FAIL: Wavefunction is not normalized.")
