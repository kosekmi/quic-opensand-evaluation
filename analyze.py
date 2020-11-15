import pandas as pd
import os
from pygnuplot import gnuplot


def get_point_type(loss):
    """
    Selects the gnuplot pointtype based on the given loss.
    :param loss:
    :return:
    """

    if loss <= 0.0001:
        return 2
    if loss <= 0.001:
        return 4
    if loss <= 0.01:
        return 8
    if loss <= 0.05:
        return 10
    return 7


def get_line_color(queue):
    """
    Selects the gnuplot linecolor based on the given queue factor.
    :param queue:
    :return:
    """

    if queue <= 1:
        return "black"
    if queue <= 2:
        return "red"
    if queue <= 5:
        return "dark-violet"
    if queue <= 10:
        return "blue"
    return "gray"


def analyze_goodput(df: pd.DataFrame, out_dir="."):
    # Save data
    df.to_pickle(os.path.join(out_dir, "goodput.pkl"))
    with open(os.path.join(out_dir, "goodput.csv"), 'w+') as out_file:
        df.to_csv(out_file)

    # Generate graphs
    for sat in df['delay'].unique():
        for rate in df['rate'].unique():
            g = gnuplot.Gnuplot(log=True)
            g.set(title='"Goodput evolution - %s - %.0f Mbit/s"' % (sat, rate),
                  key='outside right center vertical',
                  ylabel='"Goodput (kbps)"',
                  xlabel='"Time (s)"',
                  xrange='[0:30]',
                  term='pdf size 12cm, 6cm',
                  out='"%s"' % os.path.join(out_dir, "quic_goodput_%s_%s.pdf" % (sat, rate)),
                  pointsize='1',
                  logscale='y')

            # Filter only data relevant for graph
            gdf = df.loc[(df['delay'] == sat) & (df['rate'] == rate) & (df['second'] < 30)]
            gdf = gdf[['loss', 'queue', 'pep', 'second', 'bits']]
            # Calculate mean average per second over all runs
            gdf = gdf.groupby(['loss', 'queue', 'pep', 'second']).mean()

            # Collect all variations of data
            gdata = []
            for loss in df['loss'].unique():
                for queue in df['queue'].unique():
                    for pep in df['pep'].unique():
                        gdata.append((gdf.loc[(loss, queue, pep), 'bits'], pep, loss, queue))
            gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

            # Merge data to single dataframe
            plot_df = pd.concat([x[0] for x in gdata], axis=1)
            # Generate gnuplot commands
            plot_cmds = ["using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title 'QUIC%s l=%.2f%% q=%.0f'" %
                         (index + 2, get_point_type(loss), get_line_color(queue),
                          pep.upper() if pep != "none" else "", loss * 100.0, queue)
                         for index, (_, pep, loss, queue) in enumerate(gdata)]

            g.plot_data(plot_df, *plot_cmds)


def analyze_cwnd_evo(df: pd.DataFrame, out_dir="."):
    # Save data
    df.to_pickle(os.path.join(out_dir, "cwnd_evo.pkl"))
    with open(os.path.join(out_dir, "cwnd_evo.csv"), 'w+') as out_file:
        df.to_csv(out_file)

    # Generate graphs
    for sat in df['delay'].unique():
        for rate in df['rate'].unique():
            g = gnuplot.Gnuplot(log=True)
            g.set(title='"Congestion window evolution - %s - %.0f Mbit/s"' % (sat, rate),
                  key='outside right center vertical',
                  ylabel='"Congestion window (KB)"',
                  xlabel='"Time (s)"',
                  xrange='[0:30]',
                  term='pdf size 12cm, 6cm',
                  out='"%s"' % os.path.join(out_dir, "quic_cwnd_evo_%s_%s.pdf" % (sat, rate)),
                  pointsize='1',
                  logscale='y')

            # Filter only data relevant for graph
            gdf = df.loc[(df['delay'] == sat) & (df['rate'] == rate) & (df['second'] < 30)]
            gdf = gdf[['loss', 'queue', 'pep', 'second', 'cwnd']]
            # Calculate mean average per second over all runs
            gdf = gdf.groupby(['loss', 'queue', 'pep', 'second']).mean()

            # Collect all variations of data
            gdata = []
            for loss in df['loss'].unique():
                for queue in df['queue'].unique():
                    for pep in df['pep'].unique():
                        gdata.append((gdf.loc[(loss, queue, pep), 'cwnd'], pep, loss, queue))
            gdata = sorted(gdata, key=lambda x: [x[1], x[2], x[3]])

            # Merge data to single dataframe
            plot_df = pd.concat([x[0] for x in gdata], axis=1)
            # Generate gnuplot commands
            plot_cmds = ["using 1:($%d/1000) with linespoints pointtype %d linecolor '%s' title 'QUIC%s l=%.2f%% q=%.0f'" %
                         (index + 2, get_point_type(loss), get_line_color(queue),
                          pep.upper() if pep != "none" else "", loss * 100.0, queue)
                         for index, (_, pep, loss, queue) in enumerate(gdata)]

            g.plot_data(plot_df, *plot_cmds)
