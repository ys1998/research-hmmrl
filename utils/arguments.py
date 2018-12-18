"""
Commandline argument parsers.
"""
import argparse

train_parser = argparse.ArgumentParser()
train_parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the input files and \'vocab\' file')
train_parser.add_argument('--save_dir',type=str, required=True, help='Directory to store checkpointed models')
train_parser.add_argument('--best_dir',type=str, required=True, help='Directory to store best model')
train_parser.add_argument('--config_file', type=str, required=True, help='Model configuration file')
train_parser.add_argument('--job_id', type=str, required=True, help='ID of the current job')

test_parser = argparse.ArgumentParser()
test_parser.add_argument('--test_dir', type=str, help='Directory containing the input file and \'vocab\' file')
test_parser.add_argument('--prior_dir', type=str, help='Directory containing prior file with one sample per line, and vocabularies')
test_parser.add_argument('--model_dir',type=str, required=True, help='Directory where pretrained model is stored')
test_parser.add_argument('--config_file', type=str, required=True, help='Model configuration file')