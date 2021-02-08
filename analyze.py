import pandas as pd
import numpy as np
import os
import sys
import logging
from pygnuplot import gnuplot

GRAPH_DIR = 'graphs'
DATA_DIR = 'data'

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'dark-green', 'dark-orange', 'gold', 'cyan']
POINT_TYPES = [2, 4, 8, 10, 6, 12, 9, 11]

GRAPH_PLOT_SIZE_CM = (22, 8)
GRAPH_PLOT_SECONDS = 30
VALUE_PLOT_SIZE_CM = (12, 8)
MATRIX_KEY_SIZE = 0.12
MATRIX_SIZE_SKEW = 0.7

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


def get_point_type(pmap: dict, val: any):
    """
    Selects the gnuplot pointtype based on the given value. The map ensures, that the same values give the same types.
    :param pmap:
    :param val:
    :return:
    """

    if val not in pmap:
        idx = len(pmap)
        # Use default value if more point types than specified are requested
        pmap[val] = 7 if idx >= len(POINT_TYPES) else POINT_TYPES[idx]

    return pmap[val]


def get_line_color(lmap: dict, val: any):
    """
    Selects the gnuplot linecolor based on the given value. The map ensures, that the same values give the same color.
    :param lmap:
    :param val:
    :return:
    """

    if val not in lmap:
        idx = len(lmap)
        # Use default value if more line colors than specified are requested
        lmap[val] = 'gray' if idx >= len(LINE_COLORS) else LINE_COLORS[idx]

    return lmap[val]


def sat_key(sat: str):
    """
    Provides the key for sorting sat orbits from closest to earth to furthest away from earth.
    :param sat:
    :return:
    """
    try:
        return ['NONE', 'LEO', 'MEO', 'GEO'].index(sat.upper())
    except ValueError:
        return -1


def create_output_dirs(out_dir: str):
    graph_dir = os.path.join(out_dir, GRAPH_DIR)
    data_dir = os.path.join(out_dir, DATA_DIR)

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def analyze_goodput(df: pd.DataFrame, out_dir: str, extra_title_col: str = None):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    if extra_title_col is None:
        extra_title_col = 'default_extra_title'
        df[extra_title_col] = ""

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[[extra_title_col, 'protocol', 'pep', 'loss', 'second', 'bps']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby([extra_title_col, 'protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for extra_title in df[extra_title_col].unique():
                        for protocol in df['protocol'].unique():
                            for pep in df['pep'].unique():
                                for loss in df['loss'].unique():
                                    try:
                                        line_df = gdf.loc[(extra_title, protocol, pep, loss), 'bps']
                                    except KeyError:
                                        # Combination of protocol, pep and loss does not exist
                                        continue
                                    if line_df.empty:
                                        continue
                                    gdata.append((line_df, extra_title, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3], x[4]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (extra_title, protocol, pep)),
                        extra_title if len(extra_title) == 0 else ("%s " % extra_title),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, extra_title, protocol, pep, loss) in enumerate(gdata)
                ]

                g = gnuplot.Gnuplot(log=True,
                                    title='"Goodput Evolution - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Goodput (kbps)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, GRAPH_DIR,
                                                              "goodput_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')
                g.plot_data(plot_df, *plot_cmds)

                # Save plot data
                plot_df.columns = plot_cmds
                plot_df.to_csv(os.path.join(out_dir, DATA_DIR, "goodput_%s_r%s_q%d.csv" % (sat, rate, queue)))


def analyze_goodput_matrix(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / sat_cnt, 1.0 / rate_cnt)

        subfigures = []
        key_data = set()

        # Generate subfigures
        for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
            for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'bps']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for protocol in df['protocol'].unique():
                        for pep in df['pep'].unique():
                            for loss in df['loss'].unique():
                                try:
                                    line_df = gdf.loc[(protocol, pep, loss), 'bps']
                                except KeyError:
                                    # Combination of protocol, pep and loss does not exist
                                    continue
                                if line_df.empty:
                                    continue
                                gdata.append((line_df, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, protocol, pep, loss) in enumerate(gdata)
                ]

                # Add data for key
                for _, protocol, pep, loss in gdata:
                    key_data.add((protocol, pep, loss))

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    key='off',
                    ylabel='"Goodput (kbps)"',
                    xlabel='"Time (s)"',
                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                    yrange='[0:%d]' % (rate * 2000,),
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx * (1.0 - MATRIX_KEY_SIZE) / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        # Check if a matrix plot is useful
        if len(subfigures) <= 1:
            logger.debug("Skipping goodput matrix plot for q=%d, not enough individual runs", queue)
            continue

        # Null plot to add key
        key_cmds = [
            "NaN with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
            (
                get_point_type(point_map, loss),
                get_line_color(line_map, (protocol, pep)),
                protocol.upper(),
                " (PEP)" if pep else "",
                loss * 100
            )
            for protocol, pep, loss in sorted(key_data)
        ]
        subfigures.append(gnuplot.make_plot(
            *key_cmds,
            key='inside center vertical samplen 2',
            pointsize='0.5',
            size="%f, 1" % MATRIX_KEY_SIZE,
            origin="%f, 0" % (1.0 - MATRIX_KEY_SIZE),
            title=None,
            xtics=None,
            ytics=None,
            xlabel=None,
            ylabel=None,
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"Goodput Evolution - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' %
                 (GRAPH_PLOT_SIZE_CM[0] * MATRIX_SIZE_SKEW * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, GRAPH_DIR, "matrix_goodput_q%d.pdf" % queue),
        )


def analyze_cwnd_evo(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'cwnd']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for protocol in df['protocol'].unique():
                        for pep in df['pep'].unique():
                            for loss in df['loss'].unique():
                                try:
                                    line_df = gdf.loc[(protocol, pep, loss), 'cwnd']
                                except KeyError:
                                    # Combination of protocol, pep and loss does not exist
                                    continue
                                if line_df.empty:
                                    continue
                                gdata.append((line_df, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, protocol, pep, loss) in enumerate(gdata)
                ]

                g = gnuplot.Gnuplot(log=True,
                                    title='"Congestion Window Evolution - %s - %.0f Mbit/s - BDP*%d"'
                                          % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Congestion window (KB)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, GRAPH_DIR,
                                                              "cwnd_evo_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')
                g.plot_data(plot_df, *plot_cmds)

                # Save plot data
                plot_df.columns = plot_cmds
                plot_df.to_csv(os.path.join(out_dir, DATA_DIR, "cwnd_evo_%s_r%s_q%d.csv" % (sat, rate, queue)))


def analyze_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / sat_cnt, 1.0 / rate_cnt)

        subfigures = []
        key_data = set()

        # Generate subfigures
        for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
            for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'cwnd']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for protocol in df['protocol'].unique():
                        for pep in df['pep'].unique():
                            for loss in df['loss'].unique():
                                try:
                                    line_df = gdf.loc[(protocol, pep, loss), 'cwnd']
                                except KeyError:
                                    # Combination of protocol, pep and loss does not exist
                                    continue
                                if line_df.empty:
                                    continue
                                gdata.append((line_df, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, protocol, pep, loss) in enumerate(gdata)
                ]

                # Add data for key
                for _, protocol, pep, loss in gdata:
                    key_data.add((protocol, pep, loss))

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    key='off',
                    ylabel='"Congestion window (KB)"',
                    xlabel='"Time (s)"',
                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                    yrange='[0:%d]' % (rate * 3000,),
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx * (1.0 - MATRIX_KEY_SIZE) / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        # Check if a matrix plot is useful
        if len(subfigures) <= 1:
            logger.debug("Skipping congestion window evolution matrix plot for q=%d, not enough individual runs", queue)
            continue

        # Null plot to add key
        key_cmds = [
            "NaN with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
            (
                get_point_type(point_map, loss),
                get_line_color(line_map, (protocol, pep)),
                protocol.upper(),
                " (PEP)" if pep else "",
                loss * 100
            )
            for protocol, pep, loss in sorted(key_data)
        ]
        subfigures.append(gnuplot.make_plot(
            *key_cmds,
            key='inside center vertical samplen 2',
            pointsize='0.5',
            size="%f, 1" % MATRIX_KEY_SIZE,
            origin="%f, 0" % (1.0 - MATRIX_KEY_SIZE),
            title=None,
            xtics=None,
            ytics=None,
            xlabel=None,
            ylabel=None,
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"Congestion window evolution - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' %
                 (GRAPH_PLOT_SIZE_CM[0] * MATRIX_SIZE_SKEW * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, GRAPH_DIR, "matrix_cwnd_evo_q%d.pdf" % queue),
        )


def analyze_packet_loss(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                # Filter only data relevant for graph
                gdf = df.loc[
                    (df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'packets_lost']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for protocol in df['protocol'].unique():
                        for pep in df['pep'].unique():
                            for loss in df['loss'].unique():
                                try:
                                    line_df = gdf.loc[(protocol, pep, loss), 'packets_lost']
                                except KeyError:
                                    # Combination of protocol, pep and loss does not exist
                                    continue
                                if line_df.empty:
                                    continue
                                gdata.append((line_df, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:%d with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, protocol, pep, loss) in enumerate(gdata)
                ]

                g = gnuplot.Gnuplot(log=True,
                                    title='"Packet Loss - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Packets lost"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, GRAPH_DIR,
                                                              "packet_loss_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')
                g.plot_data(plot_df, *plot_cmds)

                # Save plot data
                plot_df.columns = plot_cmds
                plot_df.to_csv(os.path.join(out_dir, DATA_DIR, "packet_loss_%s_r%s_q%d.csv" % (sat, rate, queue)))


def analyze_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / sat_cnt, 1.0 / rate_cnt)

        subfigures = []
        key_data = set()

        # Generate subfigures
        for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
            ymax = df.loc[df['rate'] == rate]['packets_lost'].max()
            for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < GRAPH_PLOT_SECONDS)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'packets_lost']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for protocol in df['protocol'].unique():
                        for pep in df['pep'].unique():
                            for loss in df['loss'].unique():
                                try:
                                    line_df = gdf.loc[(protocol, pep, loss), 'packets_lost']
                                except KeyError:
                                    # Combination of protocol, pep and loss does not exist
                                    continue
                                if line_df.empty:
                                    continue
                                gdata.append((line_df, protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:%d with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        " (PEP)" if pep else "",
                        loss * 100
                    )
                    for index, (_, protocol, pep, loss) in enumerate(gdata)
                ]

                # Add data for key
                for _, protocol, pep, loss in gdata:
                    key_data.add((protocol, pep, loss))

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    ylabel='"Packets lost"',
                    xlabel='"Time (s)"',
                    xrange='[0:%d]' % GRAPH_PLOT_SECONDS,
                    yrange='[0:%d]' % (ymax,),
                    pointsize='0.5',
                    key='off',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx * (1.0 - MATRIX_KEY_SIZE) / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        # Check if a matrix plot is useful
        if len(subfigures) <= 1:
            logger.debug("Skipping packet loss matrix plot for q=%d, not enough individual runs", queue)
            continue

        # Null plot to add key
        key_cmds = [
            "NaN with linespoints pointtype %d linecolor '%s' title '%s%s l=%.2f%%'" %
            (
                get_point_type(point_map, loss),
                get_line_color(line_map, (protocol, pep)),
                protocol.upper(),
                " (PEP)" if pep else "",
                loss * 100
            )
            for protocol, pep, loss in sorted(key_data)
        ]
        subfigures.append(gnuplot.make_plot(
            *key_cmds,
            key='inside center vertical samplen 2',
            pointsize='0.5',
            size="%f, 1" % MATRIX_KEY_SIZE,
            origin="%f, 0" % (1.0 - MATRIX_KEY_SIZE),
            title=None,
            xtics=None,
            ytics=None,
            xlabel=None,
            ylabel=None,
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"Packet loss - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' %
                 (GRAPH_PLOT_SIZE_CM[0] * MATRIX_SIZE_SKEW * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, GRAPH_DIR, "matrix_packet_loss_q%d.pdf" % queue),
        )


def analyze_rtt(df: pd.DataFrame, out_dir: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                # Filter only data relevant for graph
                gdf = pd.DataFrame(df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue)])
                gdf['second'] = (gdf['seq'] / 100).astype(np.int)
                gdf = gdf[['loss', 'second', 'rtt']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
                if not gdf.empty:
                    for loss in df['loss'].unique():
                        try:
                            line_df = gdf.loc[(loss,), 'rtt']
                        except KeyError:
                            # Selected loss does not exist
                            continue
                        if line_df.empty:
                            continue
                        gdata.append((line_df, loss))
                gdata = sorted(gdata, key=lambda x: x[1])
                if len(gdata) == 0:
                    logger.debug("No data for graph (sat=%s, rate=%dmbps, queue=%d)" % (sat, rate, queue))
                    continue

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:%d with linespoints pointtype %d linecolor '%s' title 'l=%.2f%%'" %
                    (
                        index + 2,
                        get_point_type(point_map, loss),
                        get_line_color(line_map, loss),
                        loss * 100
                    )
                    for index, (_, loss) in enumerate(gdata)
                ]

                g = gnuplot.Gnuplot(log=True,
                                    title='"Round Trip Time - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"RTT (ms)"',
                                    xlabel='"Time (s)"',
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, GRAPH_DIR,
                                                              "rtt_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')
                g.plot_data(plot_df, *plot_cmds)

                # Save plot data
                plot_df.columns = plot_cmds
                plot_df.to_csv(os.path.join(out_dir, DATA_DIR, "rtt_%s_r%s_q%d.csv" % (sat, rate, queue)))


def analyze_connection_times(df: pd.DataFrame, out_dir: str, time_val: str):
    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for protocol in df['protocol'].unique():
        for pep in df['pep'].unique():
            for queue in df['queue'].unique():
                # Filter data relevant for graph
                gdf = pd.DataFrame(
                    df.loc[
                        (df['protocol'] == protocol) &
                        (df['pep'] ^ (not pep)) &
                        (df['queue'] == queue)
                        ])
                gdf = gdf[['sat', 'rate', 'loss', time_val]]
                gdf = gdf.groupby(['sat', 'rate', 'loss']).describe(percentiles=[0.05, 0.95])
                # Flatten column names
                gdf.columns = [cname for _, cname in gdf.columns]
                # Filter relevant columns
                gdf = gdf[['mean', '5%', '95%']]

                # Make sure that all combinations of sat, rate and loss exists (needed for gnuplot commands)
                # Generate a df with all combinations and NaN values, then update with real values keeping
                # NaN's where there are no data in gdf
                full_idx = pd.MultiIndex.from_product(gdf.index.levels)
                full_gdf = pd.DataFrame(index=full_idx, columns=gdf.columns)
                full_gdf.update(gdf)

                # Move index back to columns
                full_gdf.reset_index(inplace=True)

                # Generate indexes used to calculate x coordinate in plot
                sat_idx = sorted(full_gdf['sat'].unique(), key=sat_key)
                full_gdf['sat_idx'] = full_gdf['sat'].apply(lambda x: sat_idx.index(x))
                rate_idx = sorted(full_gdf['rate'].unique())
                full_gdf['rate_idx'] = full_gdf['rate'].apply(lambda x: rate_idx.index(x))
                full_gdf = full_gdf[['sat', 'sat_idx', 'rate', 'rate_idx', 'loss', 'mean', '5%', '95%']]
                full_gdf.sort_values(by=['sat_idx', 'rate_idx', 'loss'], inplace=True, ignore_index=True)

                # Create graph
                cnt_loss = len(full_gdf['loss'].unique())
                cnt_rate = len(full_gdf['rate'].unique())
                cnt_sat = len(full_gdf['sat'].unique())
                x_max = (cnt_rate + 1) * cnt_sat
                y_max = max(full_gdf['95%'].max(), full_gdf['mean'].max())
                y_max_base = 10**np.floor(np.log10(y_max))
                y_max = max(1, int(np.ceil(y_max / y_max_base) * y_max_base))

                plot_title = "Unknown"
                if time_val == 'ttfb':
                    plot_title = "Time to First Byte"
                elif time_val == 'con_est':
                    plot_title = "Connection Establishment"

                g = gnuplot.Gnuplot(log=True,
                                    title='"%s%s - BDP*%d %s"' %
                                          (protocol.upper(), " (PEP)" if pep else "", queue, plot_title),
                                    key='top left samplen 2',
                                    xlabel='"Satellite type, link capacity (Mbit/s)"',
                                    ylabel='"Time (ms)"',
                                    xrange="[0:%d]" % x_max,
                                    yrange="[0:%d]" % y_max,
                                    term="pdf size %dcm, %dcm" % VALUE_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, GRAPH_DIR, "%s_%s%s_q%d.pdf" %
                                                              (time_val, protocol, "_pep" if pep else "", queue)),
                                    pointsize='0.5')
                # Add labels for satellite types
                for sat in full_gdf['sat'].unique():
                    g.set(label='"%s" at %f,%f center' % (
                        sat.upper(),
                        (sat_idx.index(sat) + 0.5) * (cnt_rate + 1),
                        y_max * 0.075
                    ))
                # Add xtics for rates
                g.set(xtics='(%s)' % ", ".join([
                    '"%d" %d' % (rate, s_idx * (cnt_rate + 1) + rate_idx.index(rate) + 1)
                    for rate in full_gdf['rate'].unique()
                    for s_idx in range(cnt_sat)
                ]))

                plot_cmds = [
                    # using: select values for error bars (x:y:y_low:y_high)
                    "every %d::%d using ($3*%d+$5+1+%f):7:8:9 with errorbars pointtype %d linecolor '%s' title '%.2f%%'" %
                    (
                        cnt_loss,  # point increment
                        loss_idx + 1,  # start point
                        cnt_rate + 1,  # sat offset
                        (loss_idx + 1) * (0.8 / (cnt_loss + 1)) - 0.4,  # loss shift within [-0.4; +0.4]
                        get_point_type(point_map, None),
                        get_line_color(line_map, loss),
                        loss * 100
                    )
                    for loss_idx, loss in enumerate(full_gdf['loss'].unique())
                ]

                g.plot_data(full_gdf, *plot_cmds)

                # Save plot data
                full_gdf.to_csv(os.path.join(out_dir, DATA_DIR, "%s_%s%s_q%d.csv" %
                                             (time_val, protocol, "_pep" if pep else "", queue)))
                with open(os.path.join(out_dir, DATA_DIR, "%s_%s%s_q%d.gnuplot" %
                                                          (time_val, protocol, "_pep" if pep else "", queue)),
                          'w+') as f:
                    f.write("\n".join(plot_cmds))


def analyze_all(parsed_results: dict, out_dir="."):
    logger.info("Analyzing goodput")
    goodput_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'run', 'second', 'bps']
    df_goodput = pd.concat([
        parsed_results['quic_client'][goodput_cols],
        parsed_results['tcp_client'][goodput_cols],
    ], axis=0, ignore_index=True)
    analyze_goodput(df_goodput, out_dir)
    analyze_goodput_matrix(df_goodput, out_dir)

    logger.info("Analyzing congestion window evolution")
    cwnd_evo_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'run', 'second', 'cwnd']
    df_cwnd_evo = pd.concat([
        parsed_results['quic_server'][cwnd_evo_cols],
        parsed_results['tcp_server'][cwnd_evo_cols],
    ], axis=0, ignore_index=True)
    analyze_cwnd_evo(df_cwnd_evo, out_dir)
    analyze_cwnd_evo_matrix(df_cwnd_evo, out_dir)

    logger.info("Analyzing packet loss")
    pkt_loss_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'run', 'second', 'packets_lost']
    df_pkt_loss = pd.concat([
        parsed_results['quic_server'][pkt_loss_cols],
        parsed_results['tcp_server'][pkt_loss_cols],
    ], axis=0, ignore_index=True)
    analyze_packet_loss(df_pkt_loss, out_dir)
    analyze_packet_loss_matrix(df_pkt_loss, out_dir)

    logger.info("Analyzing RTT")
    rtt_cols = ['sat', 'rate', 'loss', 'queue', 'seq', 'rtt']
    df_rtt = parsed_results['ping_raw'][rtt_cols]
    analyze_rtt(df_rtt, out_dir)

    logger.info("Analyzing TTFB")
    df_con_times = pd.concat([
        parsed_results['quic_times'],
        parsed_results['tcp_times'],
    ], axis=0, ignore_index=True)
    analyze_connection_times(df_con_times, out_dir, time_val='ttfb')

    logger.info("Analyzing connection establishment")
    analyze_connection_times(df_con_times, out_dir, time_val='con_est')
