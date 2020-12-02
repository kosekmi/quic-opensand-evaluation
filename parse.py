#!/usr/bin/env python3
import os.path
import pandas as pd
import sys
import getopt
import re
import logging
import json
from enum import Enum
from analyze import analyze_all

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Mode(Enum):
    PARSE = 1
    ANALYZE = 2
    ALL = 255

    def do_parse(self):
        return self == Mode.PARSE or self == Mode.ALL

    def do_analyze(self):
        return self == Mode.ANALYZE or self == Mode.ALL


def load_json_file(path: str):
    """
    Loads and parses the content of a json file.
    :param path:
    :return:
    """

    if not os.path.isfile(path):
        return None

    with open(path, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as err:
            if err.msg != "Extra data":
                logger.exception("Failed to load json file '%s'" % path)
                return None

            # Read only first object from file, ignore extra data
            file.seek(0)
            json_str = file.read(err.pos)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.exception("Failed to read json file '%s'" % path)
                return None


def bps_factor(prefix: str):
    factor = {'K': 10e3, 'M': 10e6, 'G': 10e9, 'T': 10e12, 'P': 10e15, 'E': 10e18, 'Z': 10e21, 'Y': 10e24}
    prefix = prefix.upper()
    return factor[prefix] if prefix in factor else 1


def parse_quic_client(result_set_path):
    """
    Parse the client's output of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing QUIC client log files")
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'packets_received'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_quic_client\.txt$", file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(r"^second (\d+): (\d+(?:\.\d+)?) ([a-z]?)bit/s \((\d+) bytes received, (\d+) packets received\)$",
                                       line.strip())
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': int(line_match.group(1)),
                    'bps': int(float(line_match.group(2)) * bps_factor(line_match.group(3))),
                    'bytes': int(line_match.group(4)),
                    'packets_received': int(line_match.group(5))
                }, ignore_index=True)

    if df.empty:
        logger.warning("No QUIC client data found")

    return df


def parse_quic_server(result_set_path):
    """
    Parse the server's output of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing QUIC server log files")
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'packets_sent', 'packets_lost'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_quic_server\.txt$", file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(
                    r"^connection.*second (\d+) send window: (\d+) packets sent: (\d+) packets lost: (\d+)$",
                    line.strip())
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': int(line_match.group(1)),
                    'cwnd': int(line_match.group(2)),
                    'packets_sent': int(line_match.group(3)),
                    'packets_lost': int(line_match.group(4))
                }, ignore_index=True)

    if df.empty:
        logger.warning("No QUIC server data found")

    return df


def parse_tcp_client(result_set_path):
    """
    Parse the client's output of the TCP measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing TCP client log files")
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'omitted'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_tcp_client\.json$", file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))

        results = load_json_file(path)
        if results is None:
            logger.warning("'%s' has no content" % path)
            continue

        for interval in results['intervals']:
            df = df.append({
                'run': run,
                'second': round(interval['sum']['start']),
                'bps': float(interval['streams'][0]['bits_per_second']),
                'bytes': int(interval['streams'][0]['bytes']),
                'omitted': bool(interval['streams'][0]['omitted']),
            }, ignore_index=True)

    if df.empty:
        logger.warning("No TCP client data found")

    return df


def parse_tcp_server(result_set_path):
    """
    Parse the server's output of the TCP measurements from the log files in the given folder.
    :param result_set_path:
    :return:
    """

    logger.info("Parsing TCP server log files")
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^(\d+)_tcp_server\.json$", file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))

        results = load_json_file(path)
        if results is None:
            logger.warning("'%s' has no content" % path)
            continue

        for interval in results['intervals']:
            df = df.append({
                'run': run,
                'second': round(interval['sum']['start']),
                'cwnd': int(interval['streams'][0]['snd_cwnd']),
                'bps': float(interval['streams'][0]['bits_per_second']),
                'bytes': int(interval['streams'][0]['bytes']),
                'packets_lost': int(interval['streams'][0]['retransmits']),
                'rtt': int(interval['streams'][0]['rtt']),
                'omitted': bool(interval['streams'][0]['omitted']),
            }, ignore_index=True)

    if df.empty:
        logger.warning("No TCP server data found")

    return df


def parse_ping(result_set_path):
    logger.info("Parsing ping log files")

    path = os.path.join(result_set_path, "ping.txt")
    if not os.path.isfile(path):
        logger.warning("No ping data found")
        return None

    data = {}
    with open(path) as file:
        for line in file.readlines()[-2:]:
            match = re.search(r"^(\d+) packets transmitted, (\d+) received", line)
            if match:
                data['packets_sent'] = int(match.group(1))
                data['packets_received'] = int(match.group(2))
            else:
                match = re.search(r"= (\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) ms", line)
                if match:
                    data['rtt_min'] = float(match.group(1))
                    data['rtt_avg'] = float(match.group(2))
                    data['rtt_max'] = float(match.group(3))
                    data['rtt_mdev'] = float(match.group(4))

    return pd.DataFrame({k: [v] for k, v in data.items()})


def measure_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(path):
            logger.debug("'%s' is not a directory, skipping", folder_name)
            continue

        match = re.search(
            r"^(GEO|MEO|LEO|NONE)_r(\d+)mbit_l(\d+(?:\.\d+)?)_q(\d+(?:\.\d+)?)(?:_txq(\d+))?(?:_([a-z]+))?$",
            folder_name)
        if not match:
            logger.info("Directory '%s' doesn't match, skipping", folder_name)
            continue

        sat = match.group(1)
        rate = int(match.group(2))
        loss = float(match.group(3)) / 100.0
        queue = float(match.group(4))
        txq = int(match.group(5)) if match.group(5) else 1000
        pep = match.group(6) if match.group(6) else "none"
        yield folder_name, sat, rate, loss, queue, txq, pep


def extend_df(df, protocol, pep, sat, rate, loss, queue, txq):
    """
    Extends the dataframe containing the data of a single file with the information gained from the file path. This puts
    the single measurement in the context of all measurements.
    :param df:
    :param protocol:
    :param pep:
    :param sat:
    :param rate:
    :param loss:
    :param queue:
    :param txq:
    :return:
    """

    df['protocol'] = protocol
    df['pep'] = pep
    df['sat'] = sat
    df['rate'] = rate
    df['loss'] = loss
    df['queue'] = queue
    df['txq'] = txq
    return df


def fix_column_dtypes(df):
    """
    Ensures that the data types of each column are correct. Converts the data if necessary.
    :param df:
    :return:
    """

    numerics = {'rate', 'loss', 'queue', 'txq', 'run', 'second', 'bytes', 'cwnd', 'packets_sent', 'packets_received',
                'packets_lost', 'measured_loss', 'bps', 'rtt', 'rtt_min', 'rtt_avg', 'rtt_max', 'rtt_mdev'}
    cols = df.columns.to_list()
    for col_name in numerics.intersection(cols):
        df[col_name] = pd.to_numeric(df[col_name])

    return df


def parse(in_dir="~/measure"):
    logger.info("Parsing measurement results in '%s'", in_dir)
    df_quic_client = pd.DataFrame(columns=['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq',
                                           'run', 'second', 'bps', 'bytes', 'packets_received'])
    df_quic_server = pd.DataFrame(columns=['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq',
                                           'run', 'second', 'cwnd', 'packets_sent', 'packets_lost'])
    df_tcp_client = pd.DataFrame(columns=['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq',
                                          'run', 'second', 'bps', 'bytes', 'omitted'])
    df_tcp_server = pd.DataFrame(columns=['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq',
                                          'run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted'])
    df_ping = pd.DataFrame(columns=['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq',
                                    'packets_sent', 'packets_received', 'rtt_min', 'rtt_avg', 'rtt_max', 'rtt_mdev'])

    for folder_name, sat, rate, loss, queue, txq, pep in measure_folders(in_dir):
        logger.info("Parsing files in %s", folder_name)
        path = os.path.join(in_dir, folder_name)

        # QUIC client
        df = parse_quic_client(path)
        if df is not None:
            df = extend_df(df, 'quic', pep, sat, rate, loss, queue, txq)
            df_quic_client = df_quic_client.append(df, ignore_index=True)
        else:
            logger.warning("No data QUIC client data in %s" % folder_name)

        # QUIC server
        df = parse_quic_server(path)
        if df is not None:
            df = extend_df(df, 'quic', pep, sat, rate, loss, queue, txq)
            df_quic_server = df_quic_server.append(df, ignore_index=True)
        else:
            logger.warning("No data QUIC server data in %s" % folder_name)

        # TCP client
        df = parse_tcp_client(path)
        if df is not None:
            df = extend_df(df, 'tcp', pep, sat, rate, loss, queue, txq)
            df_tcp_client = df_tcp_client.append(df, ignore_index=True)
        else:
            logger.warning("No data TCP client data in %s" % folder_name)

        # TCP server
        df = parse_tcp_server(path)
        if df is not None:
            df = extend_df(df, 'tcp', pep, sat, rate, loss, queue, txq)
            df_tcp_server = df_tcp_server.append(df, ignore_index=True)
        else:
            logger.warning("No data TCP server data in %s" % folder_name)

        # Ping
        df = parse_ping(path)
        if df is not None:
            df = extend_df(df, 'icmp', pep, sat, rate, loss, queue, txq)
            df_ping = df_ping.append(df, ignore_index=True)
        else:
            logger.warning("No data ping data in %s" % folder_name)

    # Fix data types
    logger.info("Fixing data types")
    fix_column_dtypes(df_quic_client)
    fix_column_dtypes(df_quic_server)
    fix_column_dtypes(df_tcp_client)
    fix_column_dtypes(df_tcp_server)
    fix_column_dtypes(df_ping)

    return {
        'quic_client': df_quic_client,
        'quic_server': df_quic_server,
        'tcp_client': df_tcp_client,
        'tcp_server': df_tcp_server,
        'ping': df_ping
    }


def parse_args(argv):
    in_dir = "~/measure"
    out_dir = "."
    mode = Mode.ALL

    try:
        opts, args = getopt.getopt(argv, "i:o:pa", ["input=", "output="])
    except getopt.GetoptError:
        print("parse.py -i <inputdir> -o <outputdir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            in_dir = arg
        elif opt in ("-o", "--output"):
            out_dir = arg
        elif opt in ("-a",):
            mode = Mode.ANALYZE
        elif opt in ("-p",):
            mode = Mode.PARSE

    return in_dir, out_dir, mode


def main(argv):
    in_dir, out_dir, mode = parse_args(argv)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        logger.error("Output directory is not a directory!")
        sys.exit(1)

    parsed_results = None

    if mode.do_parse():
        logger.info("Starting parsing")
        parsed_results = parse(in_dir)
        logger.info("Parsing done")

    if parsed_results is not None:
        logger.info("Saving results")
        for name in parsed_results:
            parsed_results[name].to_pickle(os.path.join(out_dir, "%s.pkl" % name))
            with open(os.path.join(out_dir, "%s.csv" % name), 'w+') as out_file:
                parsed_results[name].to_csv(out_file)

    if mode.do_analyze():
        if parsed_results is None:
            logger.info("Loading parsed results")
            parsed_results = {}
            for file_name in os.listdir(in_dir):
                file = os.path.join(in_dir, file_name)
                if not os.path.isfile(file):
                    continue

                match = re.match(r"^(.*)\.pkl$", file_name)
                if not match:
                    continue

                logger.debug("Loading %s" % file)
                parsed_results[match.group(1)] = pd.read_pickle(file)

        logger.info("Starting analysis")
        analyze_all(parsed_results, out_dir=out_dir)
        logger.info("Analysis done")


if __name__ == '__main__':
    main(sys.argv[1:])
