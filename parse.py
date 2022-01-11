#!/usr/bin/env python3
import json
import multiprocessing as mp
import os.path
import re
import sys
from datetime import datetime
from itertools import islice
from multiprocessing.dummy.connection import Connection
from typing import Dict, List, Optional, Generator, Tuple, Callable

import numpy as np
import pandas as pd

import common
from common import logger


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


def extend_df(df: pd.DataFrame, by: pd.DataFrame, **kwargs) -> pd.DataFrame:
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


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
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
        'second': np.float32,
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
        'prime': np.float32,
    }

    # Set defaults
    df = df.fillna({col: defaults.get(dtypes[col], np.nan) for col in dtypes.keys()})

    cols = set(df.columns).intersection(dtypes.keys())
    return df.astype({col_name: dtypes[col_name] for col_name in cols})


def __mp_function_wrapper(parse_func: Callable[..., any], conn: Connection, *args, **kwargs) -> None:
    result = parse_func(*args, **kwargs)
    conn.send(result)
    conn.close()


def __parse_slice(parse_func: Callable[..., pd.DataFrame], in_dir: str, scenarios: List[Tuple[str, Dict]],
                  df_cols: List[str], protocol: str, entity: str) -> pd.DataFrame:
    """
    Parse a slice of the protocol entity results using the given function.
    :param parse_func: The function to parse a single scenario.
    :param in_dir: The directory containing the measurement results.
    :param scenarios: The scenarios to parse within the in_dir.
    :param df_cols: The column names for columns in the resulting dataframe.
    :param protocol: The name of the protocol that is being parsed.
    :param entity: Then name of the entity that is being parsed.
    :return: A dataframe containing the combined results of the specified scenarios.
    """

    df_slice = pd.DataFrame(columns=df_cols)

    for folder, config in scenarios:
        for pep in (False, True):
            df = parse_func(in_dir, folder, pep=pep)
            if df is not None:
                df_slice = extend_df(df_slice, df, protocol=protocol, pep=pep, **config)
            else:
                logger.warning("No data %s%s %s data in %s", protocol, " (pep)" if pep else "", entity, folder)

    return df_slice


def __mp_parse_slices(num_procs: int, parse_func: Callable[..., pd.DataFrame], in_dir: str,
                      scenarios: Dict[str, Dict], df_cols: List[str], protocol: str, entity: str) -> pd.DataFrame:
    """
    Parse all protocol entity results using the given function in multiple processes.
    :param num_procs: The number of processes to spawn.
    :param parse_func: The function to parse a single scenario.
    :param in_dir: The directory containing the measurement results.
    :param scenarios: The scenarios to parse within the in_dir.
    :param df_cols: The column names for columns in the resulting dataframe.
    :param protocol: The name of the protocol that is being parsed.
    :param entity: Then name of the entity that is being parsed.
    :return:
    """

    tasks = [
        (
            "%s_%s_%d" % (protocol, entity, i),
            list(islice(scenarios.items(), i, sys.maxsize, num_procs)),
            mp.Pipe()
        )
        for i in range(num_procs)
    ]
    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(__parse_slice, child_con, parse_func, in_dir, s_slice, df_cols, protocol, entity))
        for name, s_slice, (_, child_con) in tasks
    ]

    # Start processes
    for p in processes:
        p.start()

    # Collect results
    slice_dfs = [
        parent_con.recv()
        for _, _, (parent_con, _) in tasks
    ]

    # Wait for processes to finish
    for p in processes:
        p.join()

    return pd.concat(slice_dfs, axis=0, ignore_index=True)


def parse_quic_client(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                      multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all quic client results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing quic client results")
    df_cols = [*config_cols, 'run', 'second', 'bps', 'bytes', 'packets_received']
    if multi_process:
        df_quic_client = __mp_parse_slices(2, __parse_quic_client_from_scenario, in_dir, scenarios,
                                           df_cols, 'quic', 'client')
    else:
        df_quic_client = __parse_slice(__parse_quic_client_from_scenario, in_dir, [*scenarios.items()],
                                       df_cols, 'quic', 'client')

    logger.debug("Fixing quic client data types")
    df_quic_client = fix_dtypes(df_quic_client)

    logger.info("Saving quic client data")
    df_quic_client.to_pickle(os.path.join(out_dir, 'quic_client.pkl'))
    with open(os.path.join(out_dir, 'quic_client.csv'), 'w+') as out_file:
        df_quic_client.to_csv(out_file)

    return df_quic_client


def __parse_quic_client_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the quic client results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse QUIC or QUIC (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing quic%s client files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'packets_received'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        match = re.search(r"^quic%s_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))
        with open(file_path) as file:
            for line in file:
                line_match = re.search(
                    r"^second (\d+(?:\.\d+)?): (\d+(?:\.\d+)?) ([a-zA-Z]?)bit/s, bytes received: (\d+), packets received: (\d+)$",
                    line.strip()
                )
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': float(line_match.group(1)),
                    'bps': float(line_match.group(2)) * bps_factor(line_match.group(3)),
                    'bytes': int(line_match.group(4)),
                    'packets_received': int(line_match.group(5))
                }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['bps', 'bytes', 'packets_received'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No quic%s client data found", scenario_name, " (pep)" if pep else "")

    return df


def parse_quic_server(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                      multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all quic server results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing quic server results")

    df_cols = [*config_cols, 'run', 'second', 'cwnd', 'packets_sent', 'packets_lost']
    if multi_process:
        df_quic_server = __mp_parse_slices(2, __parse_quic_server_from_scenario, in_dir, scenarios,
                                           df_cols, 'quic', 'server')
    else:
        df_quic_server = __parse_slice(__parse_quic_server_from_scenario, in_dir, [*scenarios.items()],
                                       df_cols, 'quic', 'server')

    logger.debug("Fixing quic server data types")
    df_quic_server = fix_dtypes(df_quic_server)

    logger.info("Saving quic server data")
    df_quic_server.to_pickle(os.path.join(out_dir, 'quic_server.pkl'))
    with open(os.path.join(out_dir, 'quic_server.csv'), 'w+') as out_file:
        df_quic_server.to_csv(out_file)

    return df_quic_server


def __parse_quic_server_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the quic server results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse QUIC or QUIC (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing quic%s server files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'packets_sent', 'packets_lost'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(path):
            continue
        match = re.search(r"^quic%s_(\d+)_server\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))
        with open(path) as file:
            for line in file:
                line_match = re.search(
                    r"^connection \d+ second (\d+(?:\.\d+)?):.*send window: (\d+).*packets sent: (\d+).*packets lost: (\d+)$",
                    line.strip())
                if not line_match:
                    continue

                df = df.append({
                    'run': run,
                    'second': float(line_match.group(1)),
                    'cwnd': int(line_match.group(2)),
                    'packets_sent': int(line_match.group(3)),
                    'packets_lost': int(line_match.group(4))
                }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['cwnd', 'packets_sent', 'packets_lost'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No quic%s server data found", scenario_name, " (pep)" if pep else "")

    return df


def parse_quic_timing(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                      multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all quic timing results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing quic timing results")

    df_cols = [*config_cols, 'run', 'con_est', 'ttfb']
    df_quic_timing = __parse_slice(__parse_quic_timing_from_scenario, in_dir, [*scenarios.items()],
                                   df_cols, 'quic', 'timing')

    logger.debug("Fixing quic timing data types")
    df_quic_timing = fix_dtypes(df_quic_timing)

    logger.info("Saving quic timing data")
    df_quic_timing.to_pickle(os.path.join(out_dir, 'quic_timing.pkl'))
    with open(os.path.join(out_dir, 'quic_timing.csv'), 'w+') as out_file:
        df_quic_timing.to_csv(out_file)

    return df_quic_timing


def __parse_quic_timing_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the quic timing results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse QUIC or QUIC (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing quic%s timing files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'con_est', 'ttfb'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        match = re.search(r"^quic%s_ttfb_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))
        con_est = None
        ttfb = None
        with open(file_path) as file:
            for line in file:
                if line.startswith('connection establishment time:'):
                    if con_est is not None:
                        logger.warning("Found duplicate value for con_est in '%s', ignoring", file_path)
                    else:
                        con_est = float(line.split(':', 1)[1].strip()[:-2])
                elif line.startswith('time to first byte:'):
                    if ttfb is not None:
                        logger.warning("Found duplicate value for ttfb in '%s', ignoring", file_path)
                    else:
                        ttfb = float(line.split(':', 1)[1].strip()[:-2])
        df = df.append({
            'run': run,
            'con_est': con_est,
            'ttfb': ttfb
        }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['con_est', 'ttfb'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No quic%s timing data found", scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_client(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                     multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all tcp client results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing tcp client results")

    df_cols = [*config_cols, 'run', 'second', 'bps', 'bytes', 'omitted']
    if multi_process:
        df_tcp_client = __mp_parse_slices(4, __parse_tcp_client_from_scenario, in_dir, scenarios,
                                          df_cols, 'tcp', 'client')
    else:
        df_tcp_client = __parse_slice(__parse_tcp_client_from_scenario, in_dir, [*scenarios.items()],
                                      df_cols, 'tcp', 'client')

    logger.debug("Fixing tcp client data types")
    df_tcp_client = fix_dtypes(df_tcp_client)

    logger.info("Saving tcp client data")
    df_tcp_client.to_pickle(os.path.join(out_dir, 'tcp_client.pkl'))
    with open(os.path.join(out_dir, 'tcp_client.csv'), 'w+') as out_file:
        df_tcp_client.to_csv(out_file)

    return df_tcp_client


def __parse_tcp_client_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp client results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s client files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'bps', 'bytes', 'omitted'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        match = re.search(r"^tcp%s_(\d+)_client\.json$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        results = load_json_file(file_path)
        if results is None:
            logger.warning("%s: '%s' has no content", scenario_name, file_path)
            continue

        for interval in results['intervals']:
            if len(interval['streams']) == 0:
                logger.warning("%s: SKIPPING interval in '%s' due to small interval time", scenario_name, file_path)
                continue
            if float(interval['streams'][0]['seconds']) < 0.001:
                logger.warning("%s: Skipping interval in '%s' due to small interval time", scenario_name, file_path)
                continue
            df = df.append({
                'run': run,
                'second': interval['streams'][0]['end'],
                'bps': float(interval['streams'][0]['bits_per_second']),
                'bytes': int(interval['streams'][0]['bytes']),
                'omitted': bool(interval['streams'][0]['omitted']),
            }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['bps', 'bytes', 'omitted'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No tcp%s client data found", scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_server(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                     multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all tcp server results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing tcp server results")

    df_cols = [*config_cols, 'run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted']
    if multi_process:
        df_tcp_server = __mp_parse_slices(4, __parse_tcp_server_from_scenario, in_dir, scenarios,
                                          df_cols, 'tcp', 'server')
    else:
        df_tcp_server = __parse_slice(__parse_tcp_server_from_scenario, in_dir, [*scenarios.items()],
                                      df_cols, 'tcp', 'server')

    logger.debug("Fixing tcp server data types")
    df_tcp_server = fix_dtypes(df_tcp_server)

    logger.info("Saving tcp server data")
    df_tcp_server.to_pickle(os.path.join(out_dir, 'tcp_server.pkl'))
    with open(os.path.join(out_dir, 'tcp_server.csv'), 'w+') as out_file:
        df_tcp_server.to_csv(out_file)

    return df_tcp_server


def __parse_tcp_server_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp server results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s server files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'second', 'cwnd', 'bps', 'bytes', 'packets_lost', 'rtt', 'omitted'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        file_path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(file_path):
            continue
        match = re.search(r"^tcp%s_(\d+)_server\.json$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))

        results = load_json_file(file_path)
        if results is None:
            logger.warning("%s: '%s' has no content", scenario_name, file_path)
            continue

        for interval in results['intervals']:
            if len(interval['streams']) == 0:
                logger.warning("%s: SKIPPING interval in '%s' due to small interval time", scenario_name, file_path)
                continue
            if float(interval['streams'][0]['seconds']) < 0.001:
                logger.warning("%s: Skipping interval in '%s' due to small interval time", scenario_name, file_path)
                continue
            df = df.append({
                'run': run,
                'second': interval['streams'][0]['end'],
                'cwnd': int(interval['streams'][0]['snd_cwnd']),
                'bps': float(interval['streams'][0]['bits_per_second']),
                'bytes': int(interval['streams'][0]['bytes']),
                'packets_lost': int(interval['streams'][0]['retransmits']),
                'rtt': int(interval['streams'][0]['rtt']),
                'omitted': bool(interval['streams'][0]['omitted']),
            }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['bps', 'bytes', 'packets_lost'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No tcp%s server data found", scenario_name, " (pep)" if pep else "")

    return df


def parse_tcp_timing(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                     multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all tcp timing results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing tcp timing results")

    df_cols = [*config_cols, 'run', 'con_est', 'ttfb']
    df_tcp_timing = __parse_slice(__parse_tcp_timing_from_scenario, in_dir, [*scenarios.items()],
                                  df_cols, 'tcp', 'timing')

    logger.debug("Fixing tcp timing data types")
    df_tcp_timing = fix_dtypes(df_tcp_timing)

    logger.info("Saving tcp timing data")
    df_tcp_timing.to_pickle(os.path.join(out_dir, 'tcp_timing.pkl'))
    with open(os.path.join(out_dir, 'tcp_timing.csv'), 'w+') as out_file:
        df_tcp_timing.to_csv(out_file)

    return df_tcp_timing


def __parse_tcp_timing_from_scenario(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the tcp timing results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing tcp%s timing files in %s", " (pep)" if pep else "", scenario_name)
    df = pd.DataFrame(columns=['run', 'con_est', 'ttfb'])

    for file_name in os.listdir(os.path.join(in_dir, scenario_name)):
        path = os.path.join(in_dir, scenario_name, file_name)
        if not os.path.isfile(path):
            continue
        match = re.search(r"^tcp%s_ttfb_(\d+)_client\.txt$" % ("_pep" if pep else "",), file_name)
        if not match:
            continue

        logger.debug("%s: Parsing '%s'", scenario_name, file_name)
        run = int(match.group(1))
        con_est = None
        ttfb = None
        with open(path) as file:
            for line in file:
                if line.startswith('established='):
                    if con_est is not None:
                        logger.warning("%s: Found duplicate value for con_est in '%s', ignoring", scenario_name, path)
                    else:
                        con_est = float(line.split('=', 1)[1].strip()) * 1000.0
                elif line.startswith('ttfb='):
                    if ttfb is not None:
                        logger.warning("%s: Found duplicate value for ttfb in '%s', ignoring", scenario_name, path)
                    else:
                        ttfb = float(line.split('=', 1)[1].strip()) * 1000.0
        df = df.append({
            'run': run,
            'con_est': con_est,
            'ttfb': ttfb
        }, ignore_index=True)

    with_na = len(df.index)
    df.dropna(subset=['con_est', 'ttfb'], inplace=True)
    without_na = len(df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values", scenario_name, with_na - without_na)

    if df.empty:
        logger.warning("%s: No tcp%s timing data found", scenario_name, " (pep)" if pep else "")

    return df

def parse_http(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
                     multi_process: bool = False) -> pd.DataFrame:
    """
    Parse all http (selenium) timing results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: A dataframe containing the combined results from all scenarios.
    """

    logger.info("Parsing http results")

    df_cols = [*config_cols, 'run', 'domain', 'connectEnd', 'connectStart', 'responseStart', 'domInteractive', 'loadEventEnd', 'firstPaint', 'firstContentfulPaint', 'domInteractiveNorm', 'loadEventEndNorm']
    df_http = __parse_slice(__parse_http, in_dir, [*scenarios.items()],
                                df_cols, 'http', 'timing')

    if len(df_http.index) > 0:
        logger.info("Saving http timing data")
        df_http.to_pickle(os.path.join(out_dir, 'http.pkl'))
        with open(os.path.join(out_dir, 'http.csv'), 'w+') as out_file:
            df_http.to_csv(out_file)
    else:
        return None

    return df_http

def __parse_http(in_dir: str, scenario_name: str, pep: bool = False) -> pd.DataFrame:
    """
    Parse the http timing results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :param pep: Whether to parse TCP or TCP (PEP) files
    :return: A dataframe containing the parsed results of the specified scenario.
    """

    logger.debug("Parsing http%s timing files in %s", " (pep)" if pep else "", scenario_name)
    file_name='http.csv' if pep == False else 'http_pep.csv'
    file_path=os.path.join(in_dir, scenario_name, file_name)

    if not os.path.isfile(file_path):
        logger.warning(f'{file_path} was not found!')
        return None
    df = pd.read_csv(file_path, delimiter=";")
    # Filter metrics with errors out
    original_df_count = df['protocol'].count()
    df = df[df['error'].isnull()]
    df = df[df['nextHopProtocol'].notnull()]
    filtered_df_count = df['protocol'].count()
    logger.debug(f'{filtered_df_count}/{original_df_count} are valid')
    # Calculate normalized metrics
    df['domInteractiveNorm']=df.apply(lambda row: row.domInteractive - row.responseStart, axis=1)
    df['loadEventEndNorm']=df.apply(lambda row: row.loadEventEnd - row.responseStart, axis=1)
    # New normalized fields 
    df = df.reset_index()

    return df


def parse_ping(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_cols: List[str],
               multi_process: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all ping results.
    :param in_dir: The directory containing the measurement results.
    :param out_dir: The directory to save the parsed results to.
    :param scenarios: The scenarios to parse within the in_dir.
    :param config_cols: The column names for columns taken from the scenario configuration.
    :param multi_process: Whether to allow multiprocessing.
    :return: Two dataframes containing the combined results from all scenarios, one with the raw data and one with the
    summary data.
    """

    logger.info("Parsing ping results")
    df_ping_raw = pd.DataFrame(columns=[*config_cols, 'seq', 'ttl', 'rtt'])
    df_ping_summary = pd.DataFrame(columns=[*config_cols, 'packets_sent', 'packets_received', 'rtt_min',
                                            'rtt_avg', 'rtt_max', 'rtt_mdev'])

    for folder, config in scenarios.items():
        dfs = __parse_ping_from_scenario(in_dir, folder)
        if dfs is not None:
            df_ping_raw = extend_df(df_ping_raw, dfs[0], protocol='icmp', pep=False, **config)
            df_ping_summary = extend_df(df_ping_summary, dfs[1], protocol='icmp', pep=False, **config)
        else:
            logger.warning("No data ping data in %s", folder)

    logger.debug("Fixing ping data types")
    df_ping_raw = fix_dtypes(df_ping_raw)
    df_ping_summary = fix_dtypes(df_ping_summary)

    logger.info("Saving ping data")
    df_ping_raw.to_pickle(os.path.join(out_dir, 'ping_raw.pkl'))
    df_ping_summary.to_pickle(os.path.join(out_dir, 'ping_summary.pkl'))
    with open(os.path.join(out_dir, 'ping_raw.csv'), 'w+') as out_file:
        df_ping_raw.to_csv(out_file)
    with open(os.path.join(out_dir, 'ping_summary.csv'), 'w+') as out_file:
        df_ping_summary.to_csv(out_file)

    return df_ping_raw, df_ping_summary


def __parse_ping_from_scenario(in_dir: str, scenario_name: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parse the ping results in the given scenario.
    :param in_dir: The directory containing all measurement results
    :param scenario_name: The name of the scenario to parse
    :return: Two dataframes containing the parsed results of the specified scenario, one with the raw data and one with
    the summary data.
    """

    logger.debug("Parsing ping file in %s", scenario_name)

    path = os.path.join(in_dir, scenario_name, "ping.txt")
    if not os.path.isfile(path):
        logger.warning("%s: No ping data found", scenario_name)
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

    with_na = len(raw_df.index)
    raw_df.dropna(inplace=True)
    without_na = len(raw_df.index)
    if with_na != without_na:
        logger.warning("%s: Dropped %d lines with NaN values in raw data", scenario_name, with_na - without_na)

    with_na = len(summary_df.index)
    summary_df.dropna(inplace=True)
    without_na = len(summary_df.index)
    if with_na != without_na:
        logger.warning("%s, Dropped %d lines with NaN values in summary data", scenario_name, with_na - without_na)

    return raw_df, summary_df


def parse_log(in_dir: str, out_dir: str, measure_type: common.MeasureType) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_runs = None
    df_stats = None
    dfs = __parse_log(in_dir, measure_type)
    if dfs is not None:
        df_runs, df_stats = dfs
    else:
        logger.warning("No logging data")

    if df_runs is None:
        df_runs = pd.DataFrame(columns=['name'], index=pd.TimedeltaIndex([], name='time'))
    if df_stats is None:
        df_stats = pd.DataFrame(columns=['cpu_load', 'ram_usage'], index=pd.TimedeltaIndex([], name='time'))

    logger.debug("Fixing log data types")
    df_runs = fix_dtypes(df_runs)
    df_stats = fix_dtypes(df_stats)

    logger.info("Saving ping data")
    df_runs.to_pickle(os.path.join(out_dir, 'runs.pkl'))
    df_stats.to_pickle(os.path.join(out_dir, 'stats.pkl'))
    with open(os.path.join(out_dir, 'runs.csv'), 'w+') as out_file:
        df_runs.to_csv(out_file)
    with open(os.path.join(out_dir, 'stats.csv'), 'w+') as out_file:
        df_stats.to_csv(out_file)

    return df_runs, df_stats


def __parse_log(in_dir: str, measure_type: common.MeasureType) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    logger.info("Parsing log file")

    path = None
    if measure_type == common.MeasureType.OPENSAND:
        path = os.path.join(in_dir, "opensand.log")
    elif measure_type == common.MeasureType.NETEM:
        path = os.path.join(in_dir, "measure.log")
    if not os.path.isfile(path):
        logger.warning("No log file found")
        return None

    runs_data = []
    stats_data = []
    start_time = None

    with open(path) as file:
        for line in file:
            line = '00'.join(line.rsplit(':00', 1))
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


def __read_config_from_scenario(in_dir: str, scenario_name: str) -> Dict:
    config = {
        'name': scenario_name
    }

    with open(os.path.join(in_dir, scenario_name, 'config.txt'), 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    return config


def __create_config_df(out_dir: str, scenarios: Dict[str, Dict]) -> pd.DataFrame:
    df_config = pd.DataFrame(data=[config for config in scenarios.values()])
    if not df_config.empty:
        df_config.set_index('name', inplace=True)
        df_config.sort_index(inplace=True)

    logger.info("Saving config data")
    df_config.to_pickle(os.path.join(out_dir, 'config.pkl'))
    with open(os.path.join(out_dir, 'config.csv'), 'w+') as out_file:
        df_config.to_csv(out_file)

    return df_config


def parse_auto_detect(in_dir: str, out_dir: str) -> Dict:
    auto_detect = {}
    try:
        with open(os.path.join(in_dir, "environment.txt"), 'r') as env_file:
            auto_detect = {key: value for key, value in
                           filter(lambda x: len(x) == 2 and x[0] in ('MEASURE_TIME', 'REPORT_INTERVAL'),
                                  [line.split('=', 1) for line in env_file.readlines()])
                           }
    except IOError:
        pass

    with open(os.path.join(out_dir, common.AUTO_DETECT_FILE), 'w+') as out_file:
        out_file.writelines(["%s=%s" % (key, str(value)) for key, value in auto_detect.items()])

    return auto_detect


def detect_measure_type(in_dir: str, out_dir: str) -> common.MeasureType:
    logger.info("Detecting type of measurement")
    measure_type = None
    is_certain = True

    path = os.path.join(in_dir, "opensand.log")
    if os.path.isfile(path):
        if measure_type is not None:
            is_certain = False
        measure_type = common.MeasureType.OPENSAND

    path = os.path.join(in_dir, "measure.log")
    if os.path.isfile(path):
        if measure_type is not None:
            is_certain = False
        measure_type = common.MeasureType.NETEM

    if measure_type is None or not is_certain:
        logger.error("Failed to detect measurement type!")
        sys.exit(4)

    logger.info("Measure type: %s", measure_type.name)
    with open(os.path.join(out_dir, common.TYPE_FILE), 'w+') as type_file:
        type_file.write(measure_type.name)

    return measure_type


def list_result_folders(root_folder: str) -> Generator[str, None, None]:
    for folder_name in os.listdir(root_folder):
        path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(path):
            logger.debug("'%s' is not a directory, skipping", folder_name)
            continue
        yield folder_name


def __parse_results_mp(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_columns: List[str],
                       measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    tasks = [
        ('quic_client', parse_quic_client, mp.Pipe()),
        ('quic_server', parse_quic_server, mp.Pipe()),
        ('quic_timing', parse_quic_timing, mp.Pipe()),
        ('tcp_client', parse_tcp_client, mp.Pipe()),
        ('tcp_server', parse_tcp_server, mp.Pipe()),
        ('tcp_timing', parse_tcp_timing, mp.Pipe()),
        ('ping', parse_ping, mp.Pipe()),
        ('http', parse_http, mp.Pipe())
    ]

    processes = [
        mp.Process(target=__mp_function_wrapper, name=name,
                   args=(func, client_con, in_dir, out_dir, scenarios, config_columns, True))
        for name, func, (_, client_con) in tasks
    ]

    logger.info("Starting parsing processes")

    for p in processes:
        p.start()

    # work on main thread
    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    # collect data
    parsed_results = {
        name: parent_con.recv()
        for name, _, (parent_con, _) in tasks
    }

    # wait for child processes to finish
    for p in processes:
        logger.debug("Waiting for process %s", p.name)
        p.join()
    logger.info("Parsing processes done")

    parsed_results['ping_raw'], parsed_results['ping_summary'] = parsed_results['ping']
    del parsed_results['ping']

    parsed_results['config'] = df_config
    parsed_results['stats'] = df_stats
    parsed_results['runs'] = df_runs

    return parsed_results


def __parse_results_sp(in_dir: str, out_dir: str, scenarios: Dict[str, Dict], config_columns: List[str],
                       measure_type: common.MeasureType) -> Dict[str, pd.DataFrame]:
    df_quic_client = parse_quic_client(in_dir, out_dir, scenarios, config_columns)
    df_quic_server = parse_quic_server(in_dir, out_dir, scenarios, config_columns)
    df_quic_timing = parse_quic_timing(in_dir, out_dir, scenarios, config_columns)
    df_tcp_client = parse_tcp_client(in_dir, out_dir, scenarios, config_columns)
    df_tcp_server = parse_tcp_server(in_dir, out_dir, scenarios, config_columns)
    df_tcp_timing = parse_tcp_timing(in_dir, out_dir, scenarios, config_columns)
    df_http = parse_http(in_dir, out_dir, scenarios, config_columns)
    df_ping_raw, df_ping_summary = parse_ping(in_dir, out_dir, scenarios, config_columns)

    df_config = __create_config_df(out_dir, scenarios)
    df_runs, df_stats = parse_log(in_dir, out_dir, measure_type)

    return {
        'config': df_config,
        'quic_client': df_quic_client,
        'quic_server': df_quic_server,
        'quic_timing': df_quic_timing,
        'tcp_client': df_tcp_client,
        'tcp_server': df_tcp_server,
        'tcp_timing': df_tcp_timing,
        'ping_raw': df_ping_raw,
        'ping_summary': df_ping_summary,
        'stats': df_stats,
        'runs': df_runs,
        'http': df_http,
    }


def parse_results(in_dir: str, out_dir: str, multi_process: bool = False
                  ) -> Tuple[common.MeasureType, Dict, Dict[str, pd.DataFrame]]:
    logger.info("Parsing measurement results in '%s'", in_dir)

    # read scenarios
    logger.info("Reading scenarios")
    scenarios = {}
    for folder_name in list_result_folders(in_dir):
        logger.debug("Parsing config in %s", folder_name)
        scenarios[folder_name] = __read_config_from_scenario(in_dir, folder_name)

    if len(scenarios) == 0:
        print("Failed to parse results, no scenarios found")
        sys.exit(4)
    logger.info("Found %d scenarios to parse", len(scenarios))

    # create output folder
    raw_out_dir = os.path.join(out_dir, common.RAW_DATA_DIR)
    if not os.path.exists(raw_out_dir):
        os.mkdir(raw_out_dir)

    measure_type = detect_measure_type(in_dir, out_dir)
    auto_detect = parse_auto_detect(in_dir, out_dir)

    # prepare columns
    config_columns = ['protocol', 'pep', 'sat', 'prime', 'loss', 'iw']
    if measure_type == common.MeasureType.NETEM:
        config_columns.extend(['rate', 'loss', 'queue'])
    elif measure_type == common.MeasureType.OPENSAND:
        config_columns.extend(['attenuation', 'ccs', 'tbs', 'qbs', 'ubs'])

    # parse data
    parse_func = __parse_results_mp if multi_process else __parse_results_sp
    parsed_results = parse_func(in_dir, raw_out_dir, scenarios, config_columns, measure_type)

    return measure_type, auto_detect, parsed_results


def load_parsed_results(in_dir: str) -> Tuple[common.MeasureType, Dict, Dict[str, pd.DataFrame]]:
    logger.info("Loading parsed results from %s", in_dir)

    measure_type = None
    auto_detect = {}
    parsed_results = {}

    # read measure_type and auto_detect
    for file_name in os.listdir(in_dir):
        file_path = os.path.join(in_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if file_name == common.TYPE_FILE:
            logger.debug("Reading measure type")
            with open(file_path, 'r') as file:
                measure_type_str = file.readline(64)
                measure_type = common.MeasureType.from_name(measure_type_str)
                logger.debug("Read measure type str '%s' resulting in %s", measure_type_str, str(measure_type))
            continue

        if file_name == common.AUTO_DETECT_FILE:
            logger.debug("Reading auto detect file")
            with open(file_path, 'r') as file:
                auto_detect = {key: value for key, value in [line.split('=', 1) for line in file.readlines()]}

    # read data frames
    raw_dir = os.path.join(in_dir, common.RAW_DATA_DIR)
    for file_name in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        match = re.match(r"^(.*)\.pkl$", file_name)
        if not match:
            continue

        logger.debug("Loading %s", file_name)
        parsed_results[match.group(1)] = pd.read_pickle(file_path)

    return measure_type, auto_detect, parsed_results
