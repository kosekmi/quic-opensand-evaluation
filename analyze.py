import pandas as pd
import os
import sys
import logging
from pygnuplot import gnuplot

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'olive', 'dark-orange']
POINT_TYPES = [2, 4, 8, 10, 12, 6]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def get_point_type(pmap, val):
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


def get_line_color(lmap, val):
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


def analyze_goodput(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True)
                g.set(title='"Goodput evolution - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                      key='outside right center vertical',
                      ylabel='"Goodput (kbps)"',
                      xlabel='"Time (s)"',
                      xrange='[0:30]',
                      term='pdf size 18cm, 6cm',
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
                            gdata.append((gdf.loc[(protocol, pep, loss), 'bps'], protocol, pep, loss))
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


def analyze_cwnd_evo(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True)
                g.set(title='"Congestion window evolution - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                      key='outside right center vertical',
                      ylabel='"Congestion window (KB)"',
                      xlabel='"Time (s)"',
                      xrange='[0:30]',
                      term='pdf size 18cm, 6cm',
                      out='"%s"' % os.path.join(out_dir, "cwnd_evo_%s_r%s_q%d.pdf" % (sat, rate, queue)),
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


def analyze_packet_loss(df: pd.DataFrame, out_dir: str):
    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['sat'].unique():
        for rate in df['rate'].unique():
            for queue in df['queue'].unique():
                g = gnuplot.Gnuplot(log=True)
                g.set(title='"Packet loss - %s - %.0f Mbit/s - BDP*%d"' % (sat, rate, queue),
                      key='outside right center vertical',
                      ylabel='"Packets lost"',
                      xlabel='"Time (s)"',
                      xrange='[0:30]',
                      term='pdf size 18cm, 6cm',
                      out='"%s"' % os.path.join(out_dir, "packet_loss_%s_r%s_q%d.pdf" % (sat, rate, queue)),
                      pointsize='0.5')

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

                g.plot_data(plot_df, *plot_cmds)


def analyze_all(parsed_results: dict, out_dir="."):
    for k in parsed_results:
        print(k, type(parsed_results[k]), parsed_results[k].dtypes)

    logger.info("Analyzing goodput")
    goodput_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second', 'bps', 'bytes']
    df_goodput = pd.concat([
        parsed_results['quic_client'][goodput_cols],
        parsed_results['tcp_client'][goodput_cols],
    ], axis=0, ignore_index=True)
    analyze_goodput(df_goodput, out_dir)

    logger.info("Analyzing congestion window evolution")
    cwnd_evo_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second', 'cwnd']
    df_cwnd_evo = pd.concat([
        parsed_results['quic_server'][cwnd_evo_cols],
        parsed_results['tcp_server'][cwnd_evo_cols],
    ], axis=0, ignore_index=True)
    analyze_cwnd_evo(df_cwnd_evo, out_dir)

    logger.info("Analyzing packet loss")
    pkt_loss_cols = ['protocol', 'pep', 'sat', 'rate', 'loss', 'queue', 'txq', 'run', 'second', 'packets_lost']
    df_pkt_loss = pd.concat([
        parsed_results['quic_server'][pkt_loss_cols],
        parsed_results['tcp_server'][pkt_loss_cols],
    ], axis=0, ignore_index=True)
    analyze_packet_loss(df_pkt_loss, out_dir)
