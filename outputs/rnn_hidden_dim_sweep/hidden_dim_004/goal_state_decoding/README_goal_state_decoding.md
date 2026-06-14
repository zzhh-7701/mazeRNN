# Goal and state decoding

Input hidden file: outputs\rnn_hidden_dim_sweep\hidden_dim_004\hidden_analysis\hidden_states.npz
Feature space: hidden

Each decoder uses a standardized RidgeClassifierCV model.
Outputs include summary accuracy, per-class accuracy, confusion matrices, and test-set predictions.

- goal: accuracy=0.079440, balanced_accuracy=0.044230, chance=0.041100
- state: accuracy=0.102400, balanced_accuracy=0.068943, chance=0.044580
