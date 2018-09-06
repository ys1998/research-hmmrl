"""
Model specifications.

Every model must specify the following members/methods:
- Variables
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
    - __init__(config)
    - forward(sess, x, y=None, mode='train'/'val'/'test')

"""