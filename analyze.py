import pandas as pd
import numpy as np
import os
import sys
import logging
from pygnuplot import gnuplot

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'dark-green', 'dark-orange']
POINT_TYPES = [2, 4, 8, 10, 6, 12]

GRAPH_PLOT_SIZE_CM = (18, 6)
VALUE_PLOT_SIZE_CM = (12, 8)

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
        pmap[val] = 3 if idx > len(POINT_TYPES) else POINT_TYPES[idx]

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
        lmap[val] = 'gray' if idx > len(LINE_COLORS) else LINE_COLORS[idx]

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


def analyze_goodput(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True,
                                    title='"Goodput Evolution - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Goodput (kbps)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, "goodput_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')

                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'bps']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                g.plot_data(plot_df, *plot_cmds)


def analyze_goodput_matrix(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % (1.0 / sat_cnt, 1.0 / rate_cnt)

        subfigures = []

        # Generate subfigures
        for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
            for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'bps']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    key='outside right center vertical samplen 2',
                    ylabel='"Goodput (kbps)"',
                    xlabel='"Time (s)"',
                    xrange='[0:30]',
                    yrange='[0:%d]' % (rate * 2000,),
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        gnuplot.multiplot(
            *subfigures,
            title='"Goodput Evolution - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' % (GRAPH_PLOT_SIZE_CM[0] * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, "matrix_goodput_q%d.pdf" % queue),
        )


def analyze_cwnd_evo(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True,
                                    title='"Congestion Window Evolution - %s - %.0f Mbit/s - BDP*%d"'
                                          % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Congestion window (KB)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir,
                                                              "cwnd_evo_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')

                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'cwnd']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                g.plot_data(plot_df, *plot_cmds)


def analyze_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % (1.0 / sat_cnt, 1.0 / rate_cnt)

        subfigures = []

        # Generate subfigures
        for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
            for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'cwnd']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    key='outside right center vertical samplen 2',
                    ylabel='"Congestion window (KB)"',
                    xlabel='"Time (s)"',
                    xrange='[0:30]',
                    yrange='[0:%d]' % (rate * 3000,),
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        gnuplot.multiplot(
            *subfigures,
            title='"Congestion window evolution - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' % (GRAPH_PLOT_SIZE_CM[0] * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, "matrix_cwnd_evo_q%d.pdf" % queue),
        )


def analyze_packet_loss(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True,
                                    title='"Packet Loss - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Packets lost"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir,
                                                              "packet_loss_%s_r%s_q%d.pdf" % (sat, rate, queue)
                                                              ),
                                    pointsize='0.5')

                # Filter only data relevant for graph
                gdf = df.loc[
                    (df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'packets_lost']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                g.plot_data(plot_df, *plot_cmds)


def analyze_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    for queue in df['queue'].unique():
        sat_cnt = float(len(df['sat'].unique()))
        rate_cnt = float(len(df['rate'].unique()))
        sub_size = "%f, %f" % (1.0 / sat_cnt, 1.0 / rate_cnt)

        subfigures = []

        # Generate subfigures
        for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
            ymax = df.loc[df['rate'] == rate]['packets_lost'].max()
            for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
                # Filter only data relevant for graph
                gdf = df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'loss', 'second', 'packets_lost']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                sub = gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s - %.0f Mbit/s"' % (sat, rate),
                    ylabel='"Packets lost"',
                    xlabel='"Time (s)"',
                    xrange='[0:30]',
                    yrange='[0:%d]' % (ymax,),
                    pointsize='0.5',
                    key='outside right center vertical samplen 2',
                    size=sub_size,
                    origin="%f, %f" % (sat_idx / sat_cnt, rate_idx / rate_cnt)
                )
                subfigures.append(sub)

        gnuplot.multiplot(
            *subfigures,
            title='"Packet loss - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' % (GRAPH_PLOT_SIZE_CM[0] * sat_cnt, GRAPH_PLOT_SIZE_CM[1] * rate_cnt),
            output='"%s"' % os.path.join(out_dir, "matrix_packet_loss_q%d.pdf" % queue),
        )


def analyze_rtt(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True,
                                    title='"Round Trip Time - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"RTT (ms)"',
                                    xlabel='"Time (s)"',
                                    term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                                    out='"%s"' % os.path.join(out_dir, "rtt_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                                    pointsize='0.5')

                # Filter only data relevant for graph
                gdf = pd.DataFrame(df.loc[(df['sat'] == sat) & (df['rate'] == rate) & (df['queue'] == queue)])
                gdf['second'] = (gdf['seq'] / 100).astype(np.int)
                gdf = gdf[['loss', 'second', 'rtt']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['loss', 'second']).mean()

                # Collect all variations of data
                gdata = []
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

                g.plot_data(plot_df, *plot_cmds)


def analyze_connection_times(df: pd.DataFrame, out_dir: str, time_val: str):
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
                # Move index back to columns
                gdf.reset_index(inplace=True)
                gdf = gdf[['sat', 'rate', 'loss', 'mean', '5%', '95%']]

                # Make sure that all combinations of sat, rate and loss exists (needed for gnuplot commands)
                for sat in gdf['sat'].unique():
                    for rate in gdf['rate'].unique():
                        for loss in gdf['loss'].unique():
                            gdf_slice = gdf.loc[
                                (df['sat'] == sat) &
                                (df['rate'] == rate) &
                                (df['loss'] == loss)
                            ]
                            if gdf_slice.empty:
                                gdf = gdf.append({
                                    'sat': sat,
                                    'rate': rate,
                                    'loss': loss,
                                    'mean': np.nan,
                                    '5%': np.nan,
                                    '95%': np.nan
                                }, ignore_index=True)
                # Generate indexes used to calculate x coordinate in plot
                sat_idx = sorted(gdf['sat'].unique(), key=sat_key)
                gdf['sat_idx'] = gdf['sat'].apply(lambda x: sat_idx.index(x))
                rate_idx = sorted(gdf['rate'].unique())
                gdf['rate_idx'] = gdf['rate'].apply(lambda x: rate_idx.index(x))
                gdf.sort_values(by=['sat_idx', 'rate_idx', 'loss'], inplace=True, ignore_index=True)
                gdf = gdf[['sat', 'sat_idx', 'rate', 'rate_idx', 'loss', 'mean', '5%', '95%']]

                # Create graph
                cnt_loss = len(gdf['loss'].unique())
                cnt_rate = len(gdf['rate'].unique())
                cnt_sat = len(gdf['sat'].unique())
                x_max = (cnt_rate + 1) * cnt_sat
                y_max = np.ceil(gdf['95%'].max() / 1000.0) * 1000

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
                                    out='"%s"' % os.path.join(out_dir, "%s_%s%s_q%d.pdf" %
                                                              (time_val, protocol, "_pep" if pep else "", queue)),
                                    pointsize='0.5',
                                    xtics='1')
                # Add labels for satellite types
                for sat in gdf['sat'].unique():
                    g.set(label='"%s" at %f,%f center' % (
                        sat.upper(),
                        (sat_idx.index(sat) + 0.5) * (cnt_rate + 1),
                        y_max * 0.075
                    ))

                plot_cmds = [
                    # using: select values for error bars (x:y:y_low:y_high)
                    "every %d::%d using ($3*%d+$5+1+%f):7:8:9 with errorbars pointtype %d linecolor '%s' title '%.2f%%'" %
                    (
                        cnt_loss,  # point increment
                        loss_idx,  # start point
                        cnt_rate + 1,  # sat offset
                        (loss_idx + 1) * (0.8 / (cnt_loss + 1)) - 0.4,  # loss shift within [-0.4; +0.4]
                        get_point_type(point_map, loss),
                        get_line_color(line_map, loss),
                        loss * 100
                    )
                    for loss_idx, loss in enumerate(gdf['loss'].unique())
                ]
                g.plot_data(gdf, *plot_cmds)


def analyze_all(parsed_results: dict, out_dir="."):
    logger.info("Analyzing goodput")
    goodput_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second']
    df_goodput_quic = pd.DataFrame(parsed_results['quic_client'][goodput_cols])
    df_goodput_tcp = pd.DataFrame(parsed_results['tcp_client'][goodput_cols])
    # Take the most accurate value for each of the protocols
    # qperf: Byte count for each second
    # iperf: Calculated bps
    df_goodput_quic['bps'] = parsed_results['quic_client']['bytes'] * 8.0
    df_goodput_tcp['bps'] = parsed_results['tcp_client']['bps']
    df_goodput = pd.concat([df_goodput_quic, df_goodput_tcp], axis=0, ignore_index=True)
    analyze_goodput(df_goodput, out_dir)
    analyze_goodput_matrix(df_goodput, out_dir)

    logger.info("Analyzing congestion window evolution")
    cwnd_evo_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second', 'cwnd']
    df_cwnd_evo = pd.concat([
        parsed_results['quic_server'][cwnd_evo_cols],
        parsed_results['tcp_server'][cwnd_evo_cols],
    ], axis=0, ignore_index=True)
    analyze_cwnd_evo(df_cwnd_evo, out_dir)
    analyze_cwnd_evo_matrix(df_cwnd_evo, out_dir)

    logger.info("Analyzing packet loss")
    pkt_loss_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second', 'packets_lost']
    df_pkt_loss = pd.concat([
        parsed_results['quic_server'][pkt_loss_cols],
        parsed_results['tcp_server'][pkt_loss_cols],
    ], axis=0, ignore_index=True)
    analyze_packet_loss(df_pkt_loss, out_dir)
    analyze_packet_loss_matrix(df_pkt_loss, out_dir)

    logger.info("Analyzing RTT")
    rtt_cols = ['sat', 'rate', 'loss', 'queue', 'txq', 'seq', 'rtt']
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
