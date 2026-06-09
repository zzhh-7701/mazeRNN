# Hidden dynamics analysis

Input hidden file:
outputs\rnn_hidden_dim_sweep\hidden_dim_008\hidden_analysis\hidden_states.npz

This analysis defines the neural dynamics object as adjacent hidden-state pairs
within the same maze sequence:

h_t, h_(t+1), and delta_h_t = h_(t+1) - h_t.

Main outputs:
- dynamics_step_table.csv: every valid step with PCA coordinates and labels.
- hidden_transition_table.csv: adjacent transitions with delta_h and dPC values.
- hidden_update_summary.csv: mean update direction/magnitude by task, replan, action, and progress bin.
- pca_trajectory_summary.csv: mean PCA trajectory by task/replan/progress bin.
- dynamics_pca_summary.csv: PCA explained variance.
- pca_scatter_task_replan.png: sampled hidden states in PC1/PC2.
- mean_pca_trajectory_task_replan.png: mean trajectories in PC1/PC2.
- delta_norm_by_progress_task_replan.png: hidden update magnitude over trial progress.

N transitions: 170468
PC1 cumulative variance: 0.286438
PC1-3 cumulative variance: 0.588404
