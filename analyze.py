import pandas as pd
import numpy as np
import os
import sys
import logging
from pygnuplot import gnuplot
from typing import Optional, Dict, Tuple, Iterable, List, Callable, Generator
from common import Type, GRAPH_DIR, DATA_DIR

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'dark-green', 'dark-orange', 'gold', 'cyan']
POINT_TYPES = [2, 4, 8, 10, 6, 12, 9, 11]

GRAPH_PLOT_SIZE_CM = (44, 8)
GRAPH_PLOT_SECONDS = 300
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

PointMap = Dict[any, int]
LineMap = Dict[any, str]
FileTuple = Tuple[any, ...]
DataTuple = Tuple[any, ...]


def get_point_type(point_map: PointMap, val: any):
    """
    Selects the gnuplot 'pointtype' based on the given value. The map ensures, that the same values give the same types.
    :param point_map: The map to lookup point types from
    :param val: The value to lookup or generate a point type for
    :return:
    """

    if val not in point_map:
        idx = len(point_map)
        # Use default value if more point types than specified are requested
        point_map[val] = 7 if idx >= len(POINT_TYPES) else POINT_TYPES[idx]

    return point_map[val]


def get_line_color(line_map: LineMap, val: any):
    """
    Selects the gnuplot 'linecolor' based on the given value. The map ensures, that the same values give the same color.
    :param line_map: The map to lookup line colors from
    :param val: The value to lookup or generate a line color for
    :return:
    """

    if val not in line_map:
        idx = len(line_map)
        # Use default value if more line colors than specified are requested
        line_map[val] = 'gray' if idx >= len(LINE_COLORS) else LINE_COLORS[idx]

    return line_map[val]


def sat_key(sat: str):
    """
    Provides the key for sorting sat orbits from closest to earth to furthest away from earth.
    :param sat: The satellite name to sort
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


def unique_cross_product(df: pd.DataFrame, *col_names: str) -> Generator[Tuple[any, ...], None, None]:
    if len(col_names) < 1:
        yield tuple()
        return

    unique_vals = tuple(list(df[name].unique()) for name in col_names)
    vids = [0 for _ in col_names]

    while vids[0] < len(unique_vals[0]):
        yield tuple(unique_vals[cid][vid] for cid, vid in enumerate(vids))
        # Increment
        for cid in range(len(col_names) - 1, -1, -1):
            vids[cid] += 1
            if vids[cid] < len(unique_vals[cid]):
                break
            elif cid != 0:
                vids[cid] = 0


def prepare_time_series_graph_data(df: pd.DataFrame, x_col: str, y_col: str, x_range: Optional[Tuple[int, int]],
                                   y_div: float, extra_title_col: str, file_cols: List[str],
                                   file_tuple: Tuple[any, ...], data_cols: List[str], point_map: PointMap,
                                   line_map: LineMap, point_type_indices: List[int], line_color_indices: List[int],
                                   format_data_title: Callable[[Tuple[any, ...]], str]
                                   ) -> Optional[Tuple[pd.DataFrame, List[str], List[tuple]]]:
    """
    Prepare data to be used in a time series graph.

    :param df: The dataframe to read the data from
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param extra_title_col: Name of the column that holds a string prefix for the data title
    :param file_cols: Column names that define values for which separate graphs are generated
    :param file_tuple: The set of values for the file_cols that are used in this graph
    :param data_cols: Column names of the columns used for the data lines
    :param point_map: Map that ensures identical point types for same data lines
    :param line_map: Map that ensures identical line colors for same data lines
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple
    :return: A tuple consisting of a dataframe that holds all data for the graph, a list of plot commands and a list of
    data_tuples that will be plotted in the graph. If at some point there are no data left and therefore plotting the
    graph would be useless, None is returned.
    """

    # Filter data for graph
    gdf_filter = True
    if x_range is not None:
        gdf_filter = (df[x_col] >= x_range[0]) & (df[x_col] < x_range[1])
    for col_name, col_val in zip(file_cols, file_tuple):
        gdf_filter &= df[col_name] == col_val
    gdf = df.loc[gdf_filter]
    if gdf.empty:
        return None

    # Calculate mean average per y_value (e.g. per second calculate mean average from each run)
    gdf = gdf[[extra_title_col, *data_cols, x_col, y_col]]
    gdf = gdf.groupby([extra_title_col, *data_cols, x_col]).mean()

    # Calculate data lines
    gdata = []
    if not gdf.empty:
        for data_tuple in unique_cross_product(df, extra_title_col, *data_cols):
            try:
                line_df = gdf.loc[data_tuple, y_col]
            except KeyError:
                # Combination in data_tuple does not exist
                continue
            if line_df.empty:
                # Combination in data_tuple has no data
                continue
            gdata.append((line_df, data_tuple))
    gdata = sorted(gdata, key=lambda x: x[1:])
    if len(gdata) == 0:
        return None

    # Merge line data into single df
    plot_df = pd.concat([x[0] for x in gdata], axis=1)
    # Generate plot commands
    plot_cmds = [
        "using 1:($%d/%f) with linespoints pointtype %d linecolor '%s' title '%s%s'" %
        (
            index + 2,
            y_div,
            get_point_type(point_map, tuple(data_tuple[i + 1] for i in point_type_indices)),
            get_line_color(line_map, (data_tuple[0], *tuple(data_tuple[i + 1] for i in line_color_indices))),
            data_tuple[0] if len(data_tuple[0]) == 0 else ("%s " % data_tuple[0]),
            format_data_title(*data_tuple[1:])
        )
        for index, (_, data_tuple) in enumerate(gdata)
    ]

    return plot_df, plot_cmds, [data_tuple for _, data_tuple in gdata]


def plot_time_series(df: pd.DataFrame, out_dir: str, analysis_name: str, file_cols: List[str], data_cols: List[str],
                     x_col: str, y_col: str, x_range: Optional[Tuple[int, int]], y_div: float,
                     x_label: str, y_label: str, point_type_indices: List[int], line_color_indices: List[int],
                     format_data_title: Callable[[DataTuple], str], format_file_title: Callable[[FileTuple], str],
                     format_file_base: Callable[[FileTuple], str], extra_title_col: Optional[str] = None) -> None:
    """
    Plot a time series graph. It is built for, but not restricted to having a time unit (e.g. seconds) on the x-axis.

    :param df: The dataframe to read the data from
    :param out_dir: Directory where all output files are placed
    :param analysis_name: A name for the analysis used in log statements
    :param file_cols: Column names that define values for which separate graphs are generated
    :param data_cols: Column names of the columns used for the data lines
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param x_label: Label for the x-axis of the generated graphs
    :param y_label: LAbel for the y-axis of the generated graphs
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple (data_cols values)
    :param format_file_title: Function to format the title of a graph, receives a file_tuple (file_cols values)
    :param format_file_base: Function to format the base name of a graph file, receives a file_tuple (file_cols values)
    :param extra_title_col: Name of the column that holds a string prefix for the data title
    """

    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map: PointMap = {}
    line_map: LineMap = {}

    if extra_title_col is None:
        extra_title_col = 'default_extra_title'
        df[extra_title_col] = ""

    for file_tuple in unique_cross_product(df, *file_cols):
        print_file_tuple = ', '.join(["%s=%s" % (col, str(val)) for col, val in zip(file_cols, file_tuple)])
        logger.debug("Generating %s %s", analysis_name, print_file_tuple)

        prepared_data = prepare_time_series_graph_data(df,
                                                       x_col=x_col,
                                                       y_col=y_col,
                                                       x_range=x_range,
                                                       y_div=y_div,
                                                       extra_title_col=extra_title_col,
                                                       file_cols=file_cols,
                                                       file_tuple=file_tuple,
                                                       data_cols=data_cols,
                                                       point_map=point_map,
                                                       line_map=line_map,
                                                       point_type_indices=point_type_indices,
                                                       line_color_indices=line_color_indices,
                                                       format_data_title=format_data_title)
        if prepared_data is None:
            logger.debug("No data for %s %s", analysis_name, print_file_tuple)
            continue

        plot_df, plot_cmds, data_tuples = prepared_data
        file_base = format_file_base(*file_tuple)

        g = gnuplot.Gnuplot(log=True,
                            title='"%s"' % format_file_title(*file_tuple),
                            key='outside right center vertical samplen 2',
                            xlabel='"%s"' % x_label,
                            ylabel='"%s"' % y_label,
                            pointsize='0.5',
                            xrange=None,
                            yrange=None,
                            term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                            out='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, file_base))
        if x_range is not None:
            g.set(xrange='[%d:%d]' % x_range)
        g.plot_data(plot_df, *plot_cmds)

        # Save plot data
        plot_df.to_csv(os.path.join(out_dir, DATA_DIR, file_base + '.csv'))


def plot_time_series_matrix(df: pd.DataFrame, out_dir: str, analysis_name: str, file_cols: List[str],
                            data_cols: List[str], matrix_x_col: str, matrix_y_col: str, x_col: str, y_col: str,
                            x_range: Optional[Tuple[int, int]], y_div: float, x_label: str, y_label: str,
                            point_type_indices: List[int], line_color_indices: List[int],
                            format_data_title: Callable[[DataTuple], str],
                            format_file_title: Callable[[FileTuple], str],
                            format_file_base: Callable[[FileTuple], str],
                            sort_matrix_x: Callable[[Iterable], Iterable] = lambda x: sorted(x),
                            sort_matrix_y: Callable[[Iterable], Iterable] = lambda y: sorted(y),
                            extra_title_col: Optional[str] = None) -> None:
    """
    Plot multiple time series graphs arranged like a 2d-matrix based on two data values. It is built for, but not
    restricted to having a time unit (e.g. seconds) on the x-axis of each individual graph.

    :param df: The dataframe to read the data from
    :param out_dir: Directory where all output files are placed
    :param analysis_name: A name for the analysis used in log statements
    :param file_cols: Column names that define values for which separate graphs are generated
    :param data_cols: Column names of the columns used for the data lines
    :param matrix_x_col: Graphs are horizontally arranged based on values of this column
    :param matrix_y_col: Graphs are vertically arranged based on values of this column
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param x_label: Label for the x-axis of the generated graphs
    :param y_label: LAbel for the y-axis of the generated graphs
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple (data_cols values)
    :param format_file_title: Function to format the title of a graph, receives a file_tuple (file_cols values)
    :param format_file_base: Function to format the base name of a graph file, receives a file_tuple (file_cols values)
    :param sort_matrix_x: Function to sort values of the matrix_x_col, graphs will be arranged accordingly
    :param sort_matrix_y: Function to sort values of the matrix_y_col, graphs will be arranged accordingly
    :param extra_title_col: Name of the column that holds a string prefix for the data title
    """

    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map: PointMap = {}
    line_map: LineMap = {}

    if extra_title_col is None:
        extra_title_col = 'default_extra_title'
        df[extra_title_col] = ""

    for file_tuple in unique_cross_product(df, *file_cols):
        print_file_tuple = ', '.join(["%s=%s" % (col, str(val)) for col, val in zip(file_cols, file_tuple)])
        logger.debug("Generating %s matrix %s", analysis_name, print_file_tuple)

        mx_unique = df[matrix_x_col].unique()
        my_unique = df[matrix_y_col].unique()
        mx_cnt = float(len(mx_unique))
        my_cnt = float(len(my_unique))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / mx_cnt, 1.0 / my_cnt)

        subfigures = []
        key_data = set()

        # Generate subfigures
        for my_idx, my_val in enumerate(sort_matrix_y(my_unique)):
            y_max = np.ceil(df.loc[df[matrix_y_col] == my_val][y_col].max())
            for mx_idx, mx_val in enumerate(sort_matrix_x(mx_unique)):
                prepared_data = prepare_time_series_graph_data(df,
                                                               x_col=x_col,
                                                               y_col=y_col,
                                                               x_range=x_range,
                                                               y_div=y_div,
                                                               extra_title_col=extra_title_col,
                                                               file_cols=[*file_cols, matrix_x_col, matrix_y_col],
                                                               file_tuple=(*file_tuple, mx_val, my_val),
                                                               data_cols=data_cols,
                                                               point_map=point_map,
                                                               line_map=line_map,
                                                               point_type_indices=point_type_indices,
                                                               line_color_indices=line_color_indices,
                                                               format_data_title=format_data_title)
                if prepared_data is None:
                    logger.debug("No data for %s %s, %s=%s, %s=%s", analysis_name, print_file_tuple, matrix_x_col,
                                 str(mx_val), matrix_y_col, str(my_val))
                    continue

                plot_df, plot_cmds, data_tuples = prepared_data

                # Add data for key
                key_data.update(data_tuples)

                subfigures.append(gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s"' % format_file_title(*file_tuple),
                    key='off',
                    xlabel='"%s"' % x_label,
                    ylabel='"%s"' % y_label,
                    xrange=None if x_range is None else ('[%d:%d]' % x_range),
                    yrange='[0:%d]' % y_max,
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (mx_idx * (1.0 - MATRIX_KEY_SIZE) / mx_cnt, my_idx / my_cnt)
                ))

        # Check if a matrix plot is useful
        if len(subfigures) <= 1:
            logger.debug("Skipping %s matrix plot for %s, not enough individual plots", analysis_name, print_file_tuple)
            continue

        # Add null plot for key
        key_cmds = [
            "NaN with linespoints pointtype %d linecolor '%s' title '%s%s'" %
            (
                get_point_type(point_map, tuple(data_tuple[i + 1] for i in point_type_indices)),
                get_line_color(line_map, (data_tuple[0], *tuple(data_tuple[i + 1] for i in line_color_indices))),
                data_tuple[0] if len(data_tuple[0]) == 0 else ("%s " % data_tuple[0]),
                format_data_title(*data_tuple[1:])
            )
            for data_tuple in key_data
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
            xrange=None,
            yrange=None,
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"%s"' % format_file_title(*file_tuple),
            term='pdf size %dcm %dcm' %
                 (GRAPH_PLOT_SIZE_CM[0] * MATRIX_SIZE_SKEW * mx_cnt, GRAPH_PLOT_SIZE_CM[1] * my_cnt),
            output='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, format_file_base(*file_tuple)),
        )


def analyze_netem_goodput(df: pd.DataFrame, out_dir: str, extra_title_col: Optional[str] = None):
    plot_time_series(df, out_dir,
                     analysis_name='GOODPUT',
                     file_cols=['sat', 'rate', 'queue'],
                     data_cols=['protocol', 'pep', 'loss'],
                     x_col='second',
                     y_col='bps',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1000,
                     x_label="Time (s)",
                     y_label="Goodput (kbps)",
                     point_type_indices=[2],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, loss:
                     "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                     format_file_title=lambda sat, rate, queue:
                     "Goodput Evolution - %s - %.0f Mbit/s - BDP*%d" % (sat, rate, queue),
                     format_file_base=lambda sat, rate, queue:
                     "goodput_%s_r%s_q%d" % (sat, rate, queue),
                     extra_title_col=extra_title_col)


def analyze_opensand_goodput(df: pd.DataFrame, out_dir: str, extra_title_col: Optional[str] = None):
    plot_time_series(df, out_dir,
                     analysis_name='GOODPUT',
                     file_cols=['sat', 'attenuation'],
                     data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                     x_col='second',
                     y_col='bps',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1000,
                     x_label="Time (s)",
                     y_label="Goodput (kbps)",
                     point_type_indices=[2, 3, 4],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                     "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                     format_file_title=lambda sat, attenuation:
                     "Goodput Evolution - %s - %d dB" % (sat, attenuation),
                     format_file_base=lambda sat, attenuation:
                     "goodput_%s_a%d" % (sat, attenuation),
                     extra_title_col=extra_title_col)


def analyze_netem_goodput_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='GOODPUT',
                            file_cols=['queue'],
                            data_cols=['protocol', 'pep', 'loss'],
                            matrix_x_col='sat',
                            matrix_y_col='rate',
                            x_col='second',
                            y_col='bps',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1000,
                            x_label="Time (s)",
                            y_label="Goodput (kbps)",
                            point_type_indices=[2],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, loss:
                            "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                            format_file_title=lambda queue: "Goodput Evolution - BDP*%d" % queue,
                            format_file_base=lambda queue: "matrix_goodput_q%d" % queue,
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_goodput_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='GOODPUT',
                            file_cols=[],
                            data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                            matrix_x_col='sat',
                            matrix_y_col='attenuation',
                            x_col='second',
                            y_col='bps',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1000,
                            x_label="Time (s)",
                            y_label="Goodput (kbps)",
                            point_type_indices=[2, 3, 4],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                            "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                            format_file_title=lambda: "Goodput Evolution",
                            format_file_base=lambda: "matrix_goodput",
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_cwnd_evo(df: pd.DataFrame, out_dir: str):
    plot_time_series(df, out_dir,
                     analysis_name='CWND_EVO',
                     file_cols=['sat', 'rate', 'queue'],
                     data_cols=['protocol', 'pep', 'loss'],
                     x_col='second',
                     y_col='cwnd',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1000,
                     x_label="Time (s)",
                     y_label="Congestion window (KB)",
                     point_type_indices=[2],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, loss:
                     "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                     format_file_title=lambda sat, rate, queue:
                     "Congestion Window Evolution - %s - %.0f Mbit/s - BDP*%d" % (sat, rate, queue),
                     format_file_base=lambda sat, rate, queue:
                     "cwnd_evo_%s_r%s_q%d" % (sat, rate, queue))


def analyze_opensand_cwnd_evo(df: pd.DataFrame, out_dir: str):
    plot_time_series(df, out_dir,
                     analysis_name='CWND_EVO',
                     file_cols=['sat', 'attenuation'],
                     data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                     x_col='second',
                     y_col='cwnd',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1000,
                     x_label="Time (s)",
                     y_label="Congestion window (KB)",
                     point_type_indices=[2, 3, 4],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                     "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                     format_file_title=lambda sat, attenuation:
                     "Congestion Window Evolution - %s - %d dB" % (sat, attenuation),
                     format_file_base=lambda sat, attenuation:
                     "cwnd_evo_%s_a%d" % (sat, attenuation))


def analyze_netem_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='CWND_EVO',
                            file_cols=['queue'],
                            data_cols=['protocol', 'pep', 'loss'],
                            matrix_x_col='sat',
                            matrix_y_col='rate',
                            x_col='second',
                            y_col='cwnd',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1000,
                            x_label="Time (s)",
                            y_label="Congestion window (KB)",
                            point_type_indices=[2],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, loss:
                            "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                            format_file_title=lambda queue: "Congestion Window Evolution - BDP*%d" % queue,
                            format_file_base=lambda queue: "matrix_cwnd_evo_q%d" % queue,
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='CWND_EVO',
                            file_cols=[],
                            data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                            matrix_x_col='sat',
                            matrix_y_col='attenuation',
                            x_col='second',
                            y_col='cwnd',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1000,
                            x_label="Time (s)",
                            y_label="Congestion window (KB)",
                            point_type_indices=[2, 3, 4],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                            "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                            format_file_title=lambda: "Congestion Window Evolution",
                            format_file_base=lambda: "matrix_cwnd_evo",
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_packet_loss(df: pd.DataFrame, out_dir: str):
    plot_time_series(df, out_dir,
                     analysis_name='PACKET_LOSS',
                     file_cols=['sat', 'rate', 'queue'],
                     data_cols=['protocol', 'pep', 'loss'],
                     x_col='second',
                     y_col='packets_lost',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1,
                     x_label="Time (s)",
                     y_label="Packets lost",
                     point_type_indices=[2],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, loss:
                     "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                     format_file_title=lambda sat, rate, queue:
                     "Packet Loss - %s - %.0f Mbit/s - BDP*%d" % (sat, rate, queue),
                     format_file_base=lambda sat, rate, queue:
                     "packet_loss_%s_r%s_q%d" % (sat, rate, queue))


def analyze_opensand_packet_loss(df: pd.DataFrame, out_dir: str):
    plot_time_series(df, out_dir,
                     analysis_name='PACKET_LOSS',
                     file_cols=['sat', 'attenuation'],
                     data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                     x_col='second',
                     y_col='packets_lost',
                     x_range=(0, GRAPH_PLOT_SECONDS),
                     y_div=1,
                     x_label="Time (s)",
                     y_label="Packets lost",
                     point_type_indices=[2, 3, 4],
                     line_color_indices=[0, 1],
                     format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                     "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                     format_file_title=lambda sat, attenuation:
                     "Packet Loss - %s - %d dB" % (sat, attenuation),
                     format_file_base=lambda sat, attenuation:
                     "packet_loss_%s_a%d" % (sat, attenuation))


def analyze_netem_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='PACKET_LOSS',
                            file_cols=['queue'],
                            data_cols=['protocol', 'pep', 'loss'],
                            matrix_x_col='sat',
                            matrix_y_col='rate',
                            x_col='second',
                            y_col='packets_lost',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1,
                            x_label="Time (s)",
                            y_label="Packets lost",
                            point_type_indices=[2],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, loss:
                            "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                            format_file_title=lambda queue: "Packet Loss - BDP*%d" % queue,
                            format_file_base=lambda queue: "matrix_packet_loss_q%d" % queue,
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    plot_time_series_matrix(df, out_dir,
                            analysis_name='PACKET_LOSS',
                            file_cols=[],
                            data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                            matrix_x_col='sat',
                            matrix_y_col='attenuation',
                            x_col='second',
                            y_col='packets_lost',
                            x_range=(0, GRAPH_PLOT_SECONDS),
                            y_div=1000,
                            x_label="Time (s)",
                            y_label="Packets lost",
                            point_type_indices=[2, 3, 4],
                            line_color_indices=[0, 1],
                            format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                            "%s%s tbs=%s, qbs=%s, ubs=%s" % (protocol.upper(), " (PEP)" if pep else "", tbs, qbs, ubs),
                            format_file_title=lambda: "Packet Loss",
                            format_file_base=lambda: "matrix_packet_loss",
                            sort_matrix_x=lambda xvals: sorted(xvals, key=sat_key),
                            sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_rtt(df: pd.DataFrame, out_dir: str):
    df['second'] = (df['seq'] / 100).astype(np.int)
    plot_time_series(df, out_dir,
                     analysis_name='RTT',
                     file_cols=['sat', 'rate', 'queue'],
                     data_cols=['loss'],
                     x_col='second',
                     y_col='rtt',
                     x_range=None,
                     y_div=1,
                     x_label="Time (s)",
                     y_label="RTT (ms)",
                     point_type_indices=[0],
                     line_color_indices=[0],
                     format_data_title=lambda loss:
                     "l=%.2f%%" % (loss * 100),
                     format_file_title=lambda sat, rate, queue:
                     "Round Trip Time - %s - %.0f Mbit/s - BDP*%d" % (sat, rate, queue),
                     format_file_base=lambda sat, rate, queue:
                     "rtt_%s_r%s_q%d" % (sat, rate, queue))


def analyze_opensand_rtt(df: pd.DataFrame, out_dir: str):
    df['second'] = (df['seq'] / 100).astype(np.int)
    plot_time_series(df, out_dir,
                     analysis_name='RTT',
                     file_cols=['sat', 'attenuation'],
                     data_cols=[],
                     x_col='second',
                     y_col='rtt',
                     x_range=None,
                     y_div=1,
                     x_label="Time (s)",
                     y_label="RTT (ms)",
                     point_type_indices=[],
                     line_color_indices=[],
                     format_data_title=lambda: "RTT",
                     format_file_title=lambda sat, attenuation:
                     "Round Trip Time - %s - %d dB" % (sat, attenuation),
                     format_file_base=lambda sat, attenuation:
                     "rtt_%s_a%d" % (sat, attenuation))


def analyze_netem_connection_times(df: pd.DataFrame, out_dir: str, time_val: str):
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
                if gdf.empty:
                    logger.debug("No data for CON_TIMES(%s) protocol=%s, pep=%s, queue=%d", time_val, protocol, pep,
                                 queue)
                    continue

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
                y_max_base = 10 ** np.floor(np.log10(y_max))
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


def analyze_stats(df_stats, df_runs, out_dir="."):
    if df_stats.empty:
        logger.info("No stats data, skipping graph generation")
        return

    create_output_dirs(out_dir)

    def interpolate_stat(df, time, col_name):
        idx_low = df.index.get_loc(time, method='pad')
        idx_high = df.index.get_loc(time, method='backfill')

        if idx_low == idx_high:
            return df.iloc[idx_low][col_name]

        p = (time - df.index[idx_low]) / (df.index[idx_high] - df.index[idx_low])
        return df.iloc[idx_low][col_name] * (1 - p) + df.iloc[idx_high][col_name] * p

    time_max = int(df_stats.index.max() + 1)
    df_runs.reset_index(inplace=True)

    y_max = df_stats['ram_usage'].max()
    y_max_base = 10 ** np.floor(np.log10(y_max))
    y_max = max(1, int(np.ceil(y_max / y_max_base) * y_max_base))

    g = gnuplot.Gnuplot(log=True,
                        title='"RAM Usage"',
                        key='off',
                        xlabel='"Time"',
                        ylabel='"Usage (MB)"',
                        term="pdf size 44cm, 8cm",
                        xrange="[0:%d]" % time_max,
                        yrange="[0:%d]" % y_max,
                        label=df_runs.apply(
                            lambda row: '"%s" at %f,%f left rotate back textcolor \'gray\' offset 0,.5' %
                                        (
                                            row['name'], row['time'],
                                            interpolate_stat(df_stats, row['time'], 'ram_usage')),
                            axis=1).to_list(),
                        out='"%s"' % os.path.join(out_dir, GRAPH_DIR, "stats_ram.pdf"),
                        pointsize='0.5')
    plot_cmd = "using 1:2 with linespoints pointtype 2 linecolor 'black'"
    g.plot_data(df_stats[['ram_usage']], plot_cmd)

    y_max = df_stats['cpu_load'].max()
    y_max_base = 10 ** np.floor(np.log10(y_max))
    y_max = max(1, int(np.ceil(y_max / y_max_base) * y_max_base))

    g = gnuplot.Gnuplot(log=True,
                        title='"CPU Load"',
                        key='off',
                        xlabel='"Time"',
                        ylabel='"Load"',
                        xrange="[0:%d]" % time_max,
                        yrange="[0:%f]" % y_max,
                        term="pdf size 44cm, 8cm",
                        label=df_runs.apply(
                            lambda row: '"%s" at %f,%f left rotate back textcolor \'gray\' offset 0,.5' %
                                        (row['name'], row['time'], interpolate_stat(df_stats, row['time'], 'cpu_load')),
                            axis=1).to_list(),
                        out='"%s"' % os.path.join(out_dir, GRAPH_DIR, "stats_cpu.pdf"),
                        pointsize='0.5')
    plot_cmd = "using 1:2 with linespoints pointtype 2 linecolor 'black'"
    g.plot_data(df_stats[['cpu_load']], plot_cmd)


def analyze_all(parsed_results: dict, measure_type: Type, out_dir="."):
    logger.info("Analyzing goodput")
    goodput_cols = ['protocol', 'pep', 'sat', 'run', 'second', 'bps']
    if measure_type == Type.NETEM:
        goodput_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == Type.OPENSAND:
        goodput_cols.extend(['attenuation', 'tbs', 'qbs', 'ubs'])
    df_goodput = pd.concat([
        parsed_results['quic_client'][goodput_cols],
        parsed_results['tcp_client'][goodput_cols],
    ], axis=0, ignore_index=True)
    if measure_type == Type.NETEM:
        analyze_netem_goodput(df_goodput, out_dir)
        analyze_netem_goodput_matrix(df_goodput, out_dir)
    elif measure_type == Type.OPENSAND:
        analyze_opensand_goodput(df_goodput, out_dir)
        analyze_opensand_goodput_matrix(df_goodput, out_dir)

    logger.info("Analyzing congestion window evolution")
    cwnd_evo_cols = ['protocol', 'pep', 'sat', 'run', 'second', 'cwnd']
    if measure_type == Type.NETEM:
        cwnd_evo_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == Type.OPENSAND:
        cwnd_evo_cols.extend(['attenuation', 'tbs', 'qbs', 'ubs'])
    df_cwnd_evo = pd.concat([
        parsed_results['quic_server'][cwnd_evo_cols],
        parsed_results['tcp_server'][cwnd_evo_cols],
    ], axis=0, ignore_index=True)
    if measure_type == Type.NETEM:
        analyze_netem_cwnd_evo(df_cwnd_evo, out_dir)
        analyze_netem_cwnd_evo_matrix(df_cwnd_evo, out_dir)
    elif measure_type == Type.OPENSAND:
        analyze_opensand_cwnd_evo(df_cwnd_evo, out_dir)
        analyze_opensand_cwnd_evo_matrix(df_cwnd_evo, out_dir)

    logger.info("Analyzing packet loss")
    pkt_loss_cols = ['protocol', 'pep', 'sat', 'run', 'second', 'packets_lost']
    if measure_type == Type.NETEM:
        pkt_loss_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == Type.OPENSAND:
        pkt_loss_cols.extend(['attenuation', 'tbs', 'qbs', 'ubs'])
    df_pkt_loss = pd.concat([
        parsed_results['quic_server'][pkt_loss_cols],
        parsed_results['tcp_server'][pkt_loss_cols],
    ], axis=0, ignore_index=True)
    if measure_type == Type.NETEM:
        analyze_netem_packet_loss(df_pkt_loss, out_dir)
        analyze_netem_packet_loss_matrix(df_pkt_loss, out_dir)
    elif measure_type == Type.OPENSAND:
        analyze_opensand_packet_loss(df_pkt_loss, out_dir)
        analyze_opensand_packet_loss_matrix(df_pkt_loss, out_dir)

    logger.info("Analyzing RTT")
    rtt_cols = ['sat', 'seq', 'rtt']
    if measure_type == Type.NETEM:
        rtt_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == Type.OPENSAND:
        rtt_cols.extend(['attenuation', 'tbs', 'qbs', 'ubs'])
    df_rtt = parsed_results['ping_raw'][rtt_cols]
    if measure_type == Type.NETEM:
        analyze_netem_rtt(df_rtt, out_dir)
    elif measure_type == Type.OPENSAND:
        analyze_opensand_rtt(df_rtt, out_dir)

    logger.info("Analyzing TTFB")
    df_con_times = pd.concat([
        parsed_results['quic_times'],
        parsed_results['tcp_times'],
    ], axis=0, ignore_index=True)
    if measure_type == Type.NETEM:
        analyze_netem_connection_times(df_con_times, out_dir, time_val='ttfb')

    logger.info("Analyzing connection establishment")
    if measure_type == Type.NETEM:
        analyze_netem_connection_times(df_con_times, out_dir, time_val='con_est')

    logger.info("Analyzing stats")
    if measure_type == Type.OPENSAND:
        df_stats = pd.DataFrame(parsed_results['stats'])
        df_runs = pd.DataFrame(parsed_results['runs'])
        df_stats.index = df_stats.index.total_seconds()
        df_runs.index = df_runs.index.total_seconds()
        analyze_stats(df_stats, df_runs, out_dir)
