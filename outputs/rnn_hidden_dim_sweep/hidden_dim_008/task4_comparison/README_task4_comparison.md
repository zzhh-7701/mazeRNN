# Task4 comparison

This analysis compares Task1-3 against Task4 using the same hidden PCA space.

Input step table:
outputs\rnn_hidden_dim_sweep\hidden_dim_008\hidden_dynamics\dynamics_step_table.csv

Input transition table:
outputs\rnn_hidden_dim_sweep\hidden_dim_008\hidden_dynamics\hidden_transition_table.csv

Linear dynamics model:
x_(t+1) = A x_t + b

Task grouping:
- task1_3: task in 1, 2, 3
- task4: task == 4

Key fitted summaries:
- task1_3: spectral_radius=0.728357, test_r2=0.371649, attractor_like=True, fixed_point_note=ok
- task4: spectral_radius=0.896518, test_r2=0.551362, attractor_like=True, fixed_point_note=ok

Main outputs:
- task4_linear_dynamics_summary.csv
- task4_pca_trajectory_summary.csv
- task4_spectral_radius.png
- task4_fixed_points_pc1_pc2.png
- task4_pca_trajectory_pc1_pc2.png
- task4_pca_trajectory_pc1_pc2_pc3.png
