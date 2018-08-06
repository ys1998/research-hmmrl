"""Parse all the default arguments."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the input files')
parser.add_argument('--char', action='store_true', default=False, help='Train a character level RNN')
parser.add_argument('--config_file', type=str, default="config/default.yml", help='Model configuration file')
parser.add_argument('--vocab', type=str, default="vocab", help='Use SRILM processed vocabulary')
parser.add_argument('--dataset', type=str, default="ptb", help='Dataset for LM experiments')
parser.add_argument('--lm', type=str, default="LM", help='Use SRILM processed vocabulary')
parser.add_argument('--save_dir',type=str, default='save', help='Directory to store checkpointed models')
parser.add_argument('--best_dir', type=str, default='save_best', help='Directory to store best model encountered during training')
parser.add_argument('--loss_mode', type=str, default="l1", choices=["l1", "l2", "mixed", "alternate"], help='Can be l1, mixed, l2 or adaptive')
parser.add_argument('--mixed_constant', type=float, default=0.5, help='Constant for mixed loss')
parser.add_argument('--mode', type=str, default="train", choices=["test", "valid", "train", "generate"], help='train / test')
parser.add_argument('--gen_config', type=str, default="{\"prior\": \"a\", \"length\": 100}", help='Config for generate mode')
parser.add_argument('--device', type=str, default="gpu", choices=["cpu", "gpu"], help='gpu / cpu')
parser.add_argument('--job_id', type=str, required = True, help='ID of the current job')

# My additions
parser.add_argument('--T', type=float, default=1.0, help='Temperature for softmax layer')

SUMMARY = "This model uses the new augmented cost function for training"
