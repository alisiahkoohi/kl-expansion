import argparse
import json


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    parsed_args = parser.parse_args()
    parsed_args.x_range = [
        float(j) for j in parsed_args.x_range.replace('[', '').replace(
            ']', '').replace(' ', '').split(',')
    ]
    return parsed_args
