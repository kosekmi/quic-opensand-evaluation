#!/usr/bin/env python3
import os.path
import pandas as pd
import numpy as np
import sys
import re
import logging
import json
from datetime import datetime
from common import Type

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
    factor = {'K': 10 ** 3, 'M': 10 ** 6, 'G': 10 ** 9, 'T': 10 ** 12, 'P': 10 ** 15, 'E': 10 ** 18, 'Z': 10 ** 21,
              'Y': 10 ** 24}
    prefix = prefix.upper()
    return factor[prefix] if prefix in factor else 1


def parse_quic_client(result_set_path, pep=False):
    """
    Parse the client's output of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing QUIC client log files")
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'packets_received'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^quic%s_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(
                    r"^second (\d+): (\d+(?:\.\d+)?) ([a-zA-Z]?)bit/s, bytes received: (\d+), packets received: (\d+)$",
                    line.strip()
                )
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': int(line_match.group(1)),
                    'bps': float(line_match.group(2)) * bps_factor(line_match.group(3)),
                    'bytes': int(line_match.group(4)),
                    'packets_received': int(line_match.group(5))
                }, ignore_index=True)

    if df.empty:
        logger.warning("No QUIC client data found")

    return df


def parse_quic_ttfb(result_set_path, pep=False):
    """
    Parse the output of the QUIC TTFB measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing QUIC ttfb log files")
    df = pd.DataFrame(columns=['run', 'con_est', 'ttfb'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^quic%s_ttfb_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        con_est = None
        ttfb = None
        with open(path) as file:
            for line in file:
                if line.startswith('connection establishment time:'):
                    if con_est is not None:
                        logger.warning("Found duplicate value for con_est in '%s', ignoring", path)
                    else:
                        con_est = int(line.split(':', 1)[1].strip()[:-2])
                elif line.startswith('time to first byte:'):
                    if ttfb is not None:
                        logger.warning("Found duplicate value for ttfb in '%s', ignoring", path)
                    else:
                        ttfb = int(line.split(':', 1)[1].strip()[:-2])
        df = df.append({
            'run': run,
            'con_est': con_est,
            'ttfb': ttfb
        }, ignore_index=True)

    if df.empty:
        logger.warning("No QUIC ttfb data found")

    return df


def parse_quic_server(result_set_path, pep=False):
    """
    Parse the server's output of the QUIC measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing QUIC server log files")
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'packets_sent', 'packets_lost'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^quic%s_(\d+)_server\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(
                    r"^connection \d+ second (\d+):.*send window: (\d+).*packets sent: (\d+).*packets lost: (\d+)$",
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


def parse_tcp_client(result_set_path, pep=False):
    """
    Parse the client's output of the TCP measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing TCP client log files")
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'omitted'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^tcp%s_(\d+)_client\.json$" % ("_pep" if pep else "",), file_name)
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


def parse_tcp_ttfb(result_set_path, pep=False):
    """
    Parse the output of the TCP TTFB measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing TCP ttfb log files")
    df = pd.DataFrame(columns=['run', 'con_est', 'ttfb'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^tcp%s_ttfb_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("Parsing '%s'", file_name)
        run = int(match.group(1))
        con_est = None
        ttfb = None
        with open(path) as file:
            for line in file:
                if line.startswith('established='):
                    if con_est is not None:
                        logger.warning("Found duplicate value for con_est in '%s', ignoring", path)
                    else:
                        con_est = float(line.split('=', 1)[1].strip()) * 1000.0
                elif line.startswith('ttfb='):
                    if ttfb is not None:
                        logger.warning("Found duplicate value for ttfb in '%s', ignoring", path)
                    else:
                        ttfb = float(line.split('=', 1)[1].strip()) * 1000.0
        df = df.append({
            'run': run,
            'con_est': con_est,
            'ttfb': ttfb
        }, ignore_index=True)

    if df.empty:
        logger.warning("No TCP ttfb data found")

    return df


def parse_tcp_server(result_set_path, pep=False):
    """
    Parse the server's output of the TCP measurements from the log files in the given folder.
    :param result_set_path:
    :param pep:
    :return:
    """

    logger.info("Parsing TCP server log files")
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted'])

    for file_name in os.listdir(result_set_path):
        path = os.path.join(result_set_path, file_name)
        if not os.path.isfile(path):
            logger.debug("'%s' is not a file, skipping")
            continue
        match = re.search(r"^tcp%s_(\d+)_server\.json$" % ("_pep" if pep else "",), file_name)
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
    logger.info("Parsing ping log file")

    path = os.path.join(result_set_path, "ping.txt")
    if not os.path.isfile(path):
        logger.warning("No ping data found")
        return None

    raw_data = []
    summary_data = {}
    with open(path) as file:
        for line in file:
            match = re.match(r"^\d+ bytes from .*: icmp_seq=(\d+) ttl=(\d+) time=(\d+) ms", line)
            if match:
                raw_data.append({
                    'seq': match.group(1),
                    'ttl': match.group(2),
                    'rtt': match.group(3)
                })
            else:
                match = re.search(r"^(\d+) packets transmitted, (\d+) received", line)
                if match:
                    summary_data['packets_sent'] = int(match.group(1))
                    summary_data['packets_received'] = int(match.group(2))
                else:
                    match = re.search(r"= (\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) ms", line)
                    if match:
                        summary_data['rtt_min'] = float(match.group(1))
                        summary_data['rtt_avg'] = float(match.group(2))
                        summary_data['rtt_max'] = float(match.group(3))
                        summary_data['rtt_mdev'] = float(match.group(4))

    raw_df = pd.DataFrame(raw_data)
    summary_df = pd.DataFrame({k: [v] for k, v in summary_data.items()})

    return raw_df, summary_df


def parse_log(result_set_path):
    logger.info("Parsing log file")

    path = os.path.join(result_set_path, "opensand.log")
    if not os.path.isfile(path):
        path = os.path.join(result_set_path, "measure.log")
        if not os.path.isfile(path):
            logger.warning("No log file found")
            return None

    runs_data = []
    stats_data = []
    start_time = None

    with open(path) as file:
        for line in file:
            if start_time is None:
                start_time = datetime.strptime(' '.join(line.split(' ', 2)[:2]), "%Y-%m-%d %H:%M:%S%z")

            match = re.match(r"^([0-9-+ :]+) \[INFO]: (.* run \d+/\d+)$", line)
            if match:
                runs_data.append({
                    'time': datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S%z") - start_time,
                    'name': match.group(2),
                })
            else:
                match = re.search(r"^([0-9-+ :]+) \[STAT]: CPU load \(1m avg\): (\d+(?:\.\d+)?), RAM usage: (\d+)MB$",
                                  line)
                if match:
                    stats_data.append({
                        'time': datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S%z") - start_time,
                        'cpu_load': match.group(2),
                        'ram_usage': match.group(3),
                    })

    runs_df = None
    if len(runs_data) > 0:
        runs_df = pd.DataFrame(runs_data)
        runs_df.set_index('time', inplace=True)

    stats_df = None
    if len(stats_data) > 0:
        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('time', inplace=True)

    return runs_df, stats_df


def detect_measure_type(result_set_path):
    logger.info("Detecting type of measurement")

    path = os.path.join(result_set_path, "opensand.log")
    if os.path.isfile(path):
        return Type.OPENSAND

    path = os.path.join(result_set_path, "measure.log")
    if os.path.isfile(path):
        return Type.NETEM

    return None


def measure_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(path):
            logger.debug("'%s' is not a directory, skipping", folder_name)
            continue
        yield folder_name


def extend_df(df: pd.DataFrame, by: pd.DataFrame, **kwargs):
    """
    Extends the dataframe containing the data of a single file (by) with the information given in the kwargs so that it
    can be appended to the main dataframe (df)
    :param df: The main dataframe
    :param by: The dataframe to extend by
    :param kwargs: Values to use for new columns in by
    :return: The extended df
    """

    aliases = {
        'sat': ['delay', 'orbit'],
        'queue': ['queue_overhead_factor'],
    }
    missing_cols = set(df.columns).difference(set(by.columns))
    for col_name in missing_cols:
        col_value = np.nan

        if col_name in kwargs:
            col_value = kwargs[col_name]
        elif col_name in aliases:
            for alias_col in aliases[col_name]:
                if alias_col in kwargs:
                    col_value = kwargs[alias_col]
                    break

        by[col_name] = col_value
    return df.append(by, ignore_index=True)


def fix_dtypes(df):
    """
    Fix the data types of the columns in a data frame.
    :param df: The dataframe to fix
    :return:
    """

    # Cleanup values
    if 'rate' in df:
        df['rate'] = df['rate'].apply(
            lambda x: np.nan if str(x) == 'nan' else ''.join(c for c in str(x) if c.isdigit() or c == '.'))
    if 'loss' in df:
        df['loss'] = df['loss'].apply(
            lambda x: np.nan if str(x) == 'nan' else float(''.join(c for c in str(x) if c.isdigit() or c == '.')) / 100)

    defaults = {
        np.int32: -1,
        np.str: "",
        np.bool: False,
    }
    dtypes = {
        'protocol': np.str,
        'pep': np.bool,
        'sat': np.str,
        'rate': np.int32,
        'loss': float,
        'queue': np.int32,
        'run': np.int32,
        'second': np.int32,
        'bps': np.float64,
        'bytes': np.int32,
        'packets_received': np.int32,
        'cwnd': np.int32,
        'packets_sent': np.int32,
        'packets_lost': np.int32,
        'con_est': np.float64,
        'ttfb': np.float64,
        'omitted': np.bool,
        'rtt': np.int32,
        'seq': np.int32,
        'ttl': np.int32,
        'rtt_min': np.float32,
        'rtt_avg': np.float32,
        'rtt_max': np.float32,
        'rtt_mdev': np.float32,
        'name': np.str,
        'cpu_load': np.float32,
        'ram_usage': np.float32,
        'attenuation': np.int32,
        'tbs': np.str,
        'qbs': np.str,
        'ubs': np.str,
    }

    # Set defaults
    df = df.fillna({col: defaults.get(dtypes[col], np.nan) for col in dtypes.keys()})

    cols = set(df.columns).intersection(dtypes.keys())
    return df.astype({col_name: dtypes[col_name] for col_name in cols})


def parse_config(in_dir: str, folder_name: str):
    config = {
        'name': folder_name
    }

    with open(os.path.join(in_dir, folder_name, 'config.txt'), 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config


def parse(in_dir="~/measure"):
    logger.info("Parsing measurement results in '%s'", in_dir)

    measure_type = detect_measure_type(in_dir)
    if measure_type is None:
        logger.error("Failed to detect measurement type!")
        sys.exit(4)
    logger.info("Measure type: %s", measure_type.name)

    configs = []
    config_columns = ['protocol', 'pep', 'sat']
    if measure_type == Type.NETEM:
        config_columns.extend(['rate', 'loss', 'queue'])
    elif measure_type == Type.OPENSAND:
        config_columns.extend(['attenuation', 'tbs', 'qbs', 'ubs'])

    df_quic_client = pd.DataFrame(columns=[*config_columns,
                                           'run', 'second', 'bps', 'bytes', 'packets_received'])
    df_quic_server = pd.DataFrame(columns=[*config_columns,
                                           'run', 'second', 'cwnd', 'packets_sent', 'packets_lost'])
    df_quic_times = pd.DataFrame(columns=[*config_columns,
                                          'run', 'con_est', 'ttfb'])
    df_tcp_client = pd.DataFrame(columns=[*config_columns,
                                          'run', 'second', 'bps', 'bytes', 'omitted'])
    df_tcp_server = pd.DataFrame(columns=[*config_columns,
                                          'run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted'])
    df_tcp_times = pd.DataFrame(columns=[*config_columns,
                                         'run', 'con_est', 'ttfb'])
    df_ping_raw = pd.DataFrame(columns=[*config_columns,
                                        'seq', 'ttl', 'rtt'])
    df_ping_summary = pd.DataFrame(columns=[*config_columns,
                                            'packets_sent', 'packets_received', 'rtt_min', 'rtt_avg', 'rtt_max',
                                            'rtt_mdev'])

    for folder_name in measure_folders(in_dir):
        logger.info("Parsing files in %s", folder_name)
        path = os.path.join(in_dir, folder_name)

        config = parse_config(in_dir, folder_name)
        configs.append(config)

        for pep in (False, True):
            # QUIC client
            df = parse_quic_client(path, pep=pep)
            if df is not None:
                df_quic_client = extend_df(df_quic_client, df, protocol='quic', pep=pep, **config)
            else:
                logger.warning("No data QUIC%s client data in %s" % (" (PEP)" if pep else "", folder_name))

            # QUIC server
            df = parse_quic_server(path, pep=pep)
            if df is not None:
                df_quic_server = extend_df(df_quic_server, df, protocol='quic', pep=pep, **config)
            else:
                logger.warning("No data QUIC%s server data in %s" % (" (PEP)" if pep else "", folder_name))

            # QUIC ttfb
            df = parse_quic_ttfb(path, pep=pep)
            if df is not None:
                df_quic_times = extend_df(df_quic_times, df, protocol='quic', pep=pep, **config)
            else:
                logger.warning("No data QUIC%s ttfb data in %s" % (" (PEP)" if pep else "", folder_name))

            # TCP client
            df = parse_tcp_client(path, pep=pep)
            if df is not None:
                df_tcp_client = extend_df(df_tcp_client, df, protocol='tcp', pep=pep, **config)
            else:
                logger.warning("No data TCP%s client data in %s" % (" (PEP)" if pep else "", folder_name))

            # TCP server
            df = parse_tcp_server(path, pep=pep)
            if df is not None:
                df_tcp_server = extend_df(df_tcp_server, df, protocol='tcp', pep=pep, **config)
            else:
                logger.warning("No data TCP%s server data in %s" % (" (PEP)" if pep else "", folder_name))

            # TCP ttfb
            df = parse_tcp_ttfb(path, pep=pep)
            if df is not None:
                df_tcp_times = extend_df(df_tcp_times, df, protocol='tcp', pep=pep, **config)
                df_tcp_times = df_tcp_times.append(df, ignore_index=True)
            else:
                logger.warning("No data TCP%s ttfb data in %s" % (" (PEP)" if pep else "", folder_name))

        # Ping
        dfs = parse_ping(path)
        if dfs is not None:
            df_ping_raw = extend_df(df_ping_raw, dfs[0], protocol='icmp', pep=False, **config)
            df_ping_summary = extend_df(df_ping_summary, dfs[1], protocol='icmp', pep=False, **config)
        else:
            logger.warning("No data ping data in %s" % folder_name)

    df_runs = None
    df_stats = None
    dfs = parse_log(in_dir)
    if dfs is not None:
        df_runs, df_stats = dfs
    else:
        logger.warning("No logging data")

    if df_runs is None:
        df_runs = pd.DataFrame(columns=['name'], index=pd.TimedeltaIndex([], name='time'))
    if df_stats is None:
        df_stats = pd.DataFrame(columns=['cpu_load', 'ram_usage'], index=pd.TimedeltaIndex([], name='time'))

    # Fix data types
    logger.info("Fixing data types")
    df_quic_client = fix_dtypes(df_quic_client)
    df_quic_server = fix_dtypes(df_quic_server)
    df_quic_times = fix_dtypes(df_quic_times)
    df_tcp_client = fix_dtypes(df_tcp_client)
    df_tcp_server = fix_dtypes(df_tcp_server)
    df_tcp_times = fix_dtypes(df_tcp_times)
    df_ping_raw = fix_dtypes(df_ping_raw)
    df_ping_summary = fix_dtypes(df_ping_summary)
    df_runs = fix_dtypes(df_runs)
    df_stats = fix_dtypes(df_stats)

    df_config = pd.DataFrame(data=configs)
    df_config.set_index('name', inplace=True)

    return measure_type, {
        'config': df_config,
        'quic_client': df_quic_client,
        'quic_server': df_quic_server,
        'quic_times': df_quic_times,
        'tcp_client': df_tcp_client,
        'tcp_server': df_tcp_server,
        'tcp_times': df_tcp_times,
        'ping_raw': df_ping_raw,
        'ping_summary': df_ping_summary,
        'stats': df_stats,
        'runs': df_runs,
    }
