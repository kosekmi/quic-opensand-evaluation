import pandas as pd
import os
import sys
import logging
from pygnuplot import gnuplot

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'olive', 'dark-orange']
POINT_TYPES = [2, 4, 8, 10, 6, 12]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
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
                                    title='"Goodput evolution - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Goodput (kbps)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term='pdf size 12cm, 6cm',
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
                            gdata.append((
                                gdf.loc[(protocol, pep, loss), 'bps'],
                                protocol,
                                pep,
                                loss
                            ))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
                            gdata.append((
                                gdf.loc[(protocol, pep, loss), 'bps'],
                                protocol,
                                pep,
                                loss
                            ))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
            title='"Goodput evolution - BDP*%d"' % queue,
            term='pdf size %dcm, %dcm' % (12 * sat_cnt, 6 * rate_cnt),
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
                                    title='"Congestion window evolution - %s - %.0f Mbit/s - BDP*%d"'
                                          % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Congestion window (KB)"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term='pdf size 12cm, 6cm',
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
                            gdata.append((gdf.loc[(protocol, pep, loss), 'cwnd'], protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
                            gdata.append((gdf.loc[(protocol, pep, loss), 'cwnd'], protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
            term='pdf size %dcm, %dcm' % (12 * sat_cnt, 6 * rate_cnt),
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
                                    title='"Packet loss - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                                    key='outside right center vertical samplen 2',
                                    ylabel='"Packets lost"',
                                    xlabel='"Time (s)"',
                                    xrange='[0:30]',
                                    term='pdf size 12cm, 6cm',
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
                            gdata.append((gdf.loc[(protocol, pep, loss), 'packets_lost'], protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
        ymax = df['packets_lost'].max()

        subfigures = []

        # Generate subfigures
        for sat_idx, sat in enumerate(sorted(df['sat'].unique(), key=sat_key)):
            for rate_idx, rate in enumerate(sorted(df['rate'].unique(), reverse=True)):
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
                            gdata.append((gdf.loc[(protocol, pep, loss), 'packets_lost'], protocol, pep, loss))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

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
                        (" (" + pep.upper() + ")") if pep != "none" else "",
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
            term='pdf size %dcm, %dcm' % (12 * sat_cnt, 6 * rate_cnt),
            output='"%s"' % os.path.join(out_dir, "matrix_packet_loss_q%d.pdf" % queue),
        )


def analyze_all(parsed_results: dict, out_dir="."):
    logger.info("Analyzing goodput")
    goodput_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second']
    df_goodput_quic = parsed_results['quic_client'][goodput_cols]
    df_goodput_tcp = parsed_results['tcp_client'][goodput_cols]
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
