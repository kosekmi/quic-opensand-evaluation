#!/usr/bin/env python3
import getopt
import logging
import os
import re
import sys

import pandas as pd

import analyze
import parse
from common import Mode, Type, RAW_DATA_DIR


def usage(name):
    print(
        "Usage: %s -i <input> -o <output>\n"
        "\n"
        "-a, --analyze       Analyze previously parsed results\n"
        "-d, --auto-detect   Try to automatically configure analysis from input\n"
        "-h, --help          Print this help message\n"
        "-i, --input=<dir>   Input directory to read the measurement results from\n"
        "-o, --output=<dir>  Output directory to put the parsed results and graphs to\n"
        "-p, --parse         Parse only and skip analysis"
        "" % name
    )


def parse_args(name, argv):
    in_dir = "~/measure"
    out_dir = "."
    auto_detect = False
    mode = Mode.ALL

    try:
        opts, args = getopt.getopt(argv, "adhi:o:p", ["analyze", "auto-detect", "help", "input=", "output=", "parse"])
    except getopt.GetoptError:
        print("parse.py -i <input_dir> -o <output_dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-a", "analyze"):
            mode = Mode.ANALYZE
        elif opt in ("-d", "--auto-detect"):
            auto_detect = True
        elif opt in ("-h", "--help"):
            usage(name)
            sys.exit(0)
        elif opt in ("-i", "--input"):
            in_dir = arg
        elif opt in ("-o", "--output"):
            out_dir = arg
        elif opt in ("-p", "parse"):
            mode = Mode.PARSE

    return in_dir, out_dir, auto_detect, mode


def main(name, argv):
    global logger

    try:
        logger
    except NameError:
        # No logger defined
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    in_dir, out_dir, auto_detect, mode = parse_args(name, argv)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        logger.error("Output directory is not a directory!")
        sys.exit(1)

    parsed_results = None
    measure_type = None

    if mode.do_parse():
        logger.info("Starting parsing")
        measure_type, parsed_results = parse.parse(in_dir)
        logger.info("Parsing done")

    if parsed_results is not None:
        logger.info("Saving results")
        raw_dir = os.path.join(out_dir, RAW_DATA_DIR)
        if not os.path.exists(raw_dir):
            os.mkdir(raw_dir)
        if measure_type is not None:
            with open(os.path.join(raw_dir, "type"), 'w+') as type_file:
                type_file.write(measure_type.name)
        for name in parsed_results:
            parsed_results[name].to_pickle(os.path.join(raw_dir, "%s.pkl" % name))
            with open(os.path.join(raw_dir, "%s.csv" % name), 'w+') as out_file:
                parsed_results[name].to_csv(out_file)

    if mode.do_analyze():
        if parsed_results is None:
            logger.info("Loading parsed results")
            parsed_results = {}
            raw_dir = os.path.join(in_dir, RAW_DATA_DIR)
            for file_name in os.listdir(raw_dir):
                file = os.path.join(raw_dir, file_name)
                if not os.path.isfile(file):
                    continue

                if file_name == "type":
                    with open(file, 'r') as f:
                        mtype_str = f.readline(64)
                        measure_type = Type.from_name(mtype_str)
                        logger.debug("Read measure type str '%s' resulting in %s", mtype_str, str(measure_type))
                    continue

                match = re.match(r"^(.*)\.pkl$", file_name)
                if not match:
                    continue

                logger.debug("Loading %s" % file)
                parsed_results[match.group(1)] = pd.read_pickle(file)

        if auto_detect:
            env = parse.parse_env(in_dir)
            if 'MEASURE_TIME' in env:
                analyze.GRAPH_PLOT_SECONDS = float(env['MEASURE_TIME'])
                logger.debug("Detected GRAPH_PLOT_SECONDS as %f", analyze.GRAPH_PLOT_SECONDS)
            if 'REPORT_INTERVAL' in env:
                analyze.GRAPH_X_BUCKET = float(env['REPORT_INTERVAL'])
                logger.debug("Detected GRAPH_X_BUCKET as %f", analyze.GRAPH_X_BUCKET)

        logger.info("Starting analysis")
        analyze.analyze_all(parsed_results, measure_type, out_dir=out_dir)
        logger.info("Analysis done")


if __name__ == '__main__':
    main(sys.argv[0], sys.argv[1:])
