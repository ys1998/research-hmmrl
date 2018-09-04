"""
Model specifications.

Every model must specify the following members/methods:
- Placeholders
    - _input
    - _output
    - _states
    - _lr

- Other variables
    - graph
    - initial_states
    - final_states
    - train_op
    - loss
    - prediction
    - eval_metric
    - best_metric
    - global_step
    - saver

- Methods
    - __init__
    - prepare_input
    - prepare_output

"""