#!/usr/bin/env python3
import getopt
import logging
import os
import sys

import analyze
import parse
from common import Mode


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
    in_dir = None
    out_dir = None
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

    if in_dir is None:
        print("No input directory specified")
        print("%s -h for help", name)
        sys.exit(1)
    if out_dir is None:
        if mode == Mode.ANALYZE:
            out_dir = in_dir
        else:
            print("No output directory specified")
            print("%s -h for help", name)
            sys.exit(1)

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

    in_dir, out_dir, do_auto_detect, mode = parse_args(name, argv)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        logger.error("Output directory is not a directory!")
        sys.exit(1)

    measure_type = None
    auto_detect = None
    parsed_results = None

    if mode.do_parse():
        logger.info("Starting parsing")
        measure_type, auto_detect, parsed_results = parse.parse_results(in_dir, out_dir)
        logger.info("Parsing done")

    if mode.do_analyze():
        if parsed_results is None:
            measure_type, auto_detect, parsed_results = parse.load_parsed_results(in_dir)

        if do_auto_detect:
            if 'MEASURE_TIME' in auto_detect:
                analyze.GRAPH_PLOT_SECONDS = float(auto_detect['MEASURE_TIME'])
                logger.debug("Detected GRAPH_PLOT_SECONDS as %f", analyze.GRAPH_PLOT_SECONDS)
            if 'REPORT_INTERVAL' in auto_detect:
                analyze.GRAPH_X_BUCKET = float(auto_detect['REPORT_INTERVAL'])
                logger.debug("Detected GRAPH_X_BUCKET as %f", analyze.GRAPH_X_BUCKET)

        logger.info("Starting analysis")
        analyze.analyze_all(parsed_results, measure_type, out_dir=out_dir)
        logger.info("Analysis done")


if __name__ == '__main__':
    main(sys.argv[0], sys.argv[1:])
