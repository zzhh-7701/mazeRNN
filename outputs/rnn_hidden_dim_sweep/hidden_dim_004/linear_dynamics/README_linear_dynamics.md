# Linear hidden dynamics

Input transition table:
outputs\rnn_hidden_dim_sweep\hidden_dim_004\hidden_dynamics\hidden_transition_table.csv

The fitted model is an affine linear dynamical system in PCA coordinates:

x_(t+1) = A x_t + b

For each fitted condition, the script computes:
- train/test R2 for predicting x_(t+1)
- A and b coefficients
- eigenvalues of A
- spectral radius max(abs(eigenvalues))
- fixed point x* = (I - A)^(-1)b when numerically stable
- attractor-like flag, defined as spectral radius < 1

Global test R2: 0.238209
Global spectral radius: 0.588014
Global attractor-like: True

Main outputs:
- linear_dynamics_summary.csv
- fixed_points_pc1_pc2.png
- spectral_radius_by_condition.png
