#!/usr/bin/env python3
import os.path
import pandas as pd
import sys
import getopt
import re
import logging
import analyze

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def parse_quic_goodput(result_set_path):
    """
    Parse the goodput of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing QUIC goodput from log files")

    df = pd.DataFrame(columns=['run', 'second', 'bits'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_quic_goodput\.txt$", file_name)
        if not match:
            logger.debug("'%s' doesn't match, skipping", file_name)
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(r"^second (\d+):.*\((\d+) bytes received\)", line.strip())
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': int(line_match.group(1)),
                    'bits': int(line_match.group(2)) * 8
                }, ignore_index=True)

    return df


def parse_quic_cwnd_evo(result_set_path):
    """
    Parse the congestion window evolution of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing QUIC congestion window evolution from log files")

    df = pd.DataFrame(columns=['run', 'second', 'packets'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_quic_cwnd_evo\.txt$", file_name)
        if not match:
            logger.debug("'%s' doesn't match, skipping", file_name)
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(r"^connection.*second (\d+).*send window: (\d+)", line.strip())
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': int(line_match.group(1)),
                    'packets': int(line_match.group(2))
                }, ignore_index=True)

    return df


def measure_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(path):
            logger.debug("'%s' is not a directory, skipping", folder_name)
            continue

        match = re.search(r"^(GEO|MEO|LEO)_r(\d+)mbit_l(\d+(?:\.\d+)?)_q(\d+(?:\.\d+)?)(?:_([a-z]+))?$", folder_name)
        if not match:
            logger.info("Directory '%s' doesn't match, skipping", folder_name)
            continue

        delay = match.group(1)
        rate = int(match.group(2))
        loss = float(match.group(3)) / 100.0
        queue = float(match.group(4))
        pep = match.group(5) if match.group(5) else "none"
        yield folder_name, delay, rate, loss, queue, pep


def parse(in_dir="~/measure"):
    logger.info("Parsing measurement results in '%s'", in_dir)
    quic_goodput = pd.DataFrame(columns=['delay', 'rate', 'loss', 'queue', 'pep', 'run', 'second', 'bits'])
    quic_cwnd_evo = pd.DataFrame(columns=['delay', 'rate', 'loss', 'queue', 'pep', 'run', 'second', 'packets'])

    for folder_name, delay, rate, loss, queue, pep in measure_folders(in_dir):
        logger.info("Parsing files in %s", folder_name)
        path = os.path.join(in_dir, folder_name)

        # QUIC goodput
        df = parse_quic_goodput(path)
        df['delay'] = delay
        df['rate'] = rate
        df['loss'] = loss
        df['queue'] = queue
        df['pep'] = pep
        quic_goodput = quic_goodput.append(df, ignore_index=True)

        # QUIC congestion window evolution
        df = parse_quic_cwnd_evo(path)
        df['delay'] = delay
        df['rate'] = rate
        df['loss'] = loss
        df['queue'] = queue
        df['pep'] = pep
        quic_cwnd_evo = quic_cwnd_evo.append(df, ignore_index=True)

    # Fix data types
    quic_goodput['rate'] = pd.to_numeric(quic_goodput['rate'])
    quic_goodput['loss'] = pd.to_numeric(quic_goodput['loss'])
    quic_goodput['queue'] = pd.to_numeric(quic_goodput['queue'])
    quic_goodput['run'] = pd.to_numeric(quic_goodput['run'])
    quic_goodput['second'] = pd.to_numeric(quic_goodput['second'])
    quic_goodput['bits'] = pd.to_numeric(quic_goodput['bits'])

    quic_cwnd_evo['rate'] = pd.to_numeric(quic_cwnd_evo['rate'])
    quic_cwnd_evo['loss'] = pd.to_numeric(quic_cwnd_evo['loss'])
    quic_cwnd_evo['queue'] = pd.to_numeric(quic_cwnd_evo['queue'])
    quic_cwnd_evo['run'] = pd.to_numeric(quic_cwnd_evo['run'])
    quic_cwnd_evo['second'] = pd.to_numeric(quic_cwnd_evo['second'])
    quic_cwnd_evo['packets'] = pd.to_numeric(quic_cwnd_evo['packets'])

    return quic_goodput, quic_cwnd_evo


def main(argv):
    in_dir = "~/measure"
    out_dir = "."

    try:
        opts, args = getopt.getopt(argv, "i:o:", ["input=", "output="])
    except getopt.GetoptError:
        print("parse.py -i <inputdir> -o <outputdir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            in_dir = arg
        elif opt in ("-o", "--output"):
            out_dir = arg

    quic_goodput, quic_cwnd_evo = parse(in_dir)
    analyze.analyze_quic_goodput(quic_goodput, out_dir=out_dir)
    analyze.analyze_quic_cwnd_evo(quic_cwnd_evo, out_dir=out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
