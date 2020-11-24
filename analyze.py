import pandas as pd
import os
from pygnuplot import gnuplot

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'olive', 'dark-orange']
POINT_TYPES = [2, 4, 8, 10, 12, 6]


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


def analyze_goodput(df: pd.DataFrame, out_dir="."):
    # Save data
    df.to_pickle(os.path.join(out_dir, "goodput.pkl"))
    with open(os.path.join(out_dir, "goodput.csv"), 'w+') as out_file:
        df.to_csv(out_file)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['delay'].unique():
        for rate in df['rate'].unique():
            for loss in df['loss'].unique():
                g = gnuplot.Gnuplot(log=True)
                g.set(title='"Goodput evolution - %s - %.0f Mbit/s - %.2f%%"' % (sat, rate, loss * 100),
                      key='outside right center vertical',
                      ylabel='"Goodput (kbps)"',
                      xlabel='"Time (s)"',
                      xrange='[0:30]',
                      term='pdf size 18cm, 6cm',
                      out='"%s"' % os.path.join(out_dir, "goodput_%s_r%s_l%.2f.pdf" % (sat, rate, loss * 100)),
                      pointsize='0.5')

                # Filter only data relevant for graph
                gdf = df.loc[(df['delay'] == sat) & (df['rate'] == rate) & (df['loss'] == loss) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'txq', 'second', 'bits']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'txq', 'second']).mean()

                # Collect all variations of data
                gdata = []
                for protocol in df['protocol'].unique():
                    for pep in df['pep'].unique():
                        for txq in df['txq'].unique():
                            gdata.append((gdf.loc[(protocol, pep, txq), 'bits'], protocol, pep, txq))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s txq=%d'" %
                    (
                        index + 2,
                        get_point_type(point_map, txq),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        (" (" + pep.upper() + ")") if pep != "none" else "",
                        txq
                    )
                    for index, (_, protocol, pep, txq) in enumerate(gdata)
                ]

                g.plot_data(plot_df, *plot_cmds)


def analyze_cwnd_evo(df: pd.DataFrame, out_dir="."):
    # Save data
    df.to_pickle(os.path.join(out_dir, "cwnd_evo.pkl"))
    with open(os.path.join(out_dir, "cwnd_evo.csv"), 'w+') as out_file:
        df.to_csv(out_file)

    # Ensures same point types and line colors across all graphs
    point_map = {}
    line_map = {}

    # Generate graphs
    for sat in df['delay'].unique():
        for rate in df['rate'].unique():
            for loss in df['loss'].unique():
                g = gnuplot.Gnuplot(log=True)
                g.set(title='"Congestion window evolution - %s - %.0f Mbit/s - %.2f%%"' % (sat, rate, loss * 100),
                      key='outside right center vertical',
                      ylabel='"Congestion window (KB)"',
                      xlabel='"Time (s)"',
                      xrange='[0:30]',
                      term='pdf size 18cm, 6cm',
                      out='"%s"' % os.path.join(out_dir, "cwnd_evo_%s_r%s_l%.2f.pdf" % (sat, rate, loss * 100)),
                      pointsize='0.5')

                # Filter only data relevant for graph
                gdf = df.loc[(df['delay'] == sat) & (df['rate'] == rate) & (df['loss'] == loss) & (df['second'] < 30)]
                gdf = gdf[['protocol', 'pep', 'txq', 'second', 'cwnd']]
                # Calculate mean average per second over all runs
                gdf = gdf.groupby(['protocol', 'pep', 'txq', 'second']).mean()

                # Collect all variations of data
                gdata = []
                for protocol in df['protocol'].unique():
                    for pep in df['pep'].unique():
                        for txq in df['txq'].unique():
                            gdata.append((gdf.loc[(protocol, pep, txq), 'cwnd'], protocol, pep, txq))
                gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

                # Merge data to single dataframe
                plot_df = pd.concat([x[0] for x in gdata], axis=1)
                # Generate gnuplot commands
                plot_cmds = [
                    "using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title '%s%s txq=%d'" %
                    (
                        index + 2,
                        get_point_type(point_map, txq),
                        get_line_color(line_map, (protocol, pep)),
                        protocol.upper(),
                        (" (" + pep.upper() + ")") if pep != "none" else "",
                        txq
                    )
                    for index, (_, protocol, pep, txq) in enumerate(gdata)
                ]

                g.plot_data(plot_df, *plot_cmds)
