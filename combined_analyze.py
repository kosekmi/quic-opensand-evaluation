#!/usr/bin/env python3
import os.path
import pandas as pd
import sys
import re
import getopt
import analyze
from parse import RAW_DATA_DIR


def analyze_quic_goodput(out_dir: str, data: list):
    quic_data = [(title, results['quic_client']) for title, _, results in data]

    # Add title to data
    for title, df in quic_data:
        df['combined_title'] = title

    # Combine all dataframes
    df_quic_goodput = pd.concat([df for _, df in quic_data], axis=0, ignore_index=True)
    analyze.analyze_goodput(df_quic_goodput, out_dir, extra_title_col='combined_title')


def parse_args(argv):
    out_dir = "."

    try:
        opts, args = getopt.getopt(argv, "o:", ["output="])
    except getopt.GetoptError:
        print("parse.py -i <input_dir> -o <output_dir>")
        sys.exit(2)

    # Parse options
    for opt, arg in opts:
        if opt in ("-o", "--output"):
            out_dir = arg

    # Parse arguments
    if len(args) < 4 or len(args) % 2 != 0:
        print("Invalid number of arguments: <title1> <path1> <title2> <path2> [... <titleN> <pathN>]")

    data = []
    for i in range(0, len(args), 2):
        data.append((args[i], args[i + 1]))

    return out_dir, data


def main(argv):
    out_dir, data = parse_args(argv)

    # Load parsed data
    def load_data(path: str):
        parsed_results = {}
        raw_dir = os.path.join(path, RAW_DATA_DIR)
        for file_name in os.listdir(raw_dir):
            file = os.path.join(raw_dir, file_name)
            if not os.path.isfile(file):
                continue

            match = re.match(r"^(.*)\.pkl$", file_name)
            if not match:
                continue

            parsed_results[match.group(1)] = pd.read_pickle(file)

        return parsed_results

    data = [(title, path, load_data(path)) for title, path in data]

    analyze_quic_goodput(out_dir, data)


if __name__ == '__main__':
    main(sys.argv[1:])
