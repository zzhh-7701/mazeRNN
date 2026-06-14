# Goal and state decoding

Input hidden file: outputs\rnn_hidden_dim_sweep\hidden_dim_008\hidden_analysis\hidden_states.npz
Feature space: hidden

Each decoder uses a standardized RidgeClassifierCV model.
Outputs include summary accuracy, per-class accuracy, confusion matrices, and test-set predictions.

- goal: accuracy=0.086880, balanced_accuracy=0.051922, chance=0.041100
- state: accuracy=0.251760, balanced_accuracy=0.171686, chance=0.044580
