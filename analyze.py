import logging
import multiprocessing as mp
import os
import sys
from typing import Optional, Dict, Tuple, Iterable, List, Callable, Generator

import numpy as np
import pandas as pd
from pygnuplot import gnuplot

from common import MeasureType, GRAPH_DIR, DATA_DIR

LINE_COLORS = ['black', 'red', 'dark-violet', 'blue', 'dark-green', 'dark-orange', 'gold', 'cyan', 'spring-green',
               'orange', 'greenyellow', 'violet', 'royalblue', 'dark-pink', 'pink', 'seagreen']
POINT_TYPES = [2, 4, 8, 10, 6, 12, 9, 11, 13, 15, 17, 20, 22, 33, 34, 50]

GRAPH_PLOT_SIZE_CM = (44, 8)
GRAPH_PLOT_SECONDS = 30
GRAPH_PLOT_RTT_SECONDS = 100
GRAPH_X_BUCKET = 0.1
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
    handler.setFormatter(logging.Formatter('%(asctime)s %(processName)s [%(levelname)s] %(message)s'))
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


def sat_tuple_key(section_tuple: Tuple[any, ...]):
    return sat_key(section_tuple[0])


def create_output_dirs(out_dir: str):
    graph_dir = os.path.join(out_dir, GRAPH_DIR)
    data_dir = os.path.join(out_dir, DATA_DIR)

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def sprint_tuple(col_names: List[str], col_values: Tuple[any, ...]) -> str:
    return ', '.join(["%s=%s" % (col, str(val)) for col, val in zip(col_names, col_values)])


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


def filter_graph_data(df: pd.DataFrame, x_col: str, x_range: Optional[Tuple[int, int]], file_cols: List[str],
                      file_tuple: FileTuple) -> Optional[pd.DataFrame]:
    """
    Filter data relevant for the graph from the dataframe.
    :param df: The dataframe to filter
    :param x_col: Name of the column that has the data for the x-axis, only used if x_range is given
    :param x_range: (min, max) tuple for filtering the values for the x-axis, or None for no filter
    :param file_cols: Column names that define values for which separate graphs are generated
    :param file_tuple: The set of values for the file_cols that are used in this graph
    :return:
    """

    gdf_filter = True

    if x_range is not None:
        gdf_filter = (df[x_col] >= x_range[0]) & (df[x_col] < x_range[1])

    for col_name, col_val in zip(file_cols, file_tuple):
        gdf_filter &= df[col_name] == col_val

    gdf = df.loc[gdf_filter]
    return None if gdf.empty else gdf


def prepare_time_series_graph_data(df: pd.DataFrame, x_col: str, y_col: str, x_range: Optional[Tuple[int, int]],
                                   x_bucket: Optional[float], y_div: float, extra_title_col: str, file_cols: List[str],
                                   file_tuple: FileTuple, data_cols: List[str], point_map: PointMap, line_map: LineMap,
                                   point_type_indices: List[int], line_color_indices: List[int],
                                   format_data_title: Callable[[DataTuple], str]
                                   ) -> Optional[Tuple[pd.DataFrame, List[str], List[tuple]]]:
    """
    Prepare data to be used in a time series graph.

    :param df: The dataframe to read the data from
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param x_bucket: Size of the bucket to use for aggregating data on the x-axis
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
    gdf = filter_graph_data(df, x_col, x_range, file_cols, file_tuple)
    if gdf is None or gdf.empty:
        return None
    gdf = pd.DataFrame(gdf)

    if x_bucket is not None:
        if x_range is not None:
            start, end = x_range
        else:
            start = gdf[x_col].min()
            end = gdf[x_col].max()
        # Start one bucket earlier to add zero data point (lines start at origin)
        # End one bucket after since each bucket is defined as [a;b) with a being the name of the bucket
        buckets = np.arange(start=start - x_bucket, stop=end + x_bucket, step=x_bucket)
        gdf[x_col] = pd.cut(gdf[x_col], buckets, labels=buckets[1:])

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
            if line_df.empty or line_df.isnull().values.all():
                # Combination in data_tuple has no data
                continue
            gdata.append((line_df, data_tuple))
    gdata = sorted(gdata, key=lambda x: x[1:])
    if len(gdata) == 0:
        return None

    # Merge line data into single df
    plot_df = pd.concat([x[0] for x in gdata], axis=1)
    # Make first category (named 0.0) start at the origin
    plot_df.iloc[0] = 0
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
                     x_col: str, y_col: str, x_range: Optional[Tuple[int, int]], x_bucket: Optional[float],
                     y_div: float, x_label: str, y_label: str, point_type_indices: List[int],
                     line_color_indices: List[int], format_data_title: Callable[[DataTuple], str],
                     format_file_title: Callable[[FileTuple], str], format_file_base: Callable[[FileTuple], str],
                     extra_title_col: Optional[str] = None) -> None:
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
    :param x_bucket: Size of the bucket to use for aggregating data on the x-axis
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
        print_file_tuple = sprint_tuple(file_cols, file_tuple)
        logger.debug("Generating %s %s", analysis_name, print_file_tuple)

        prepared_data = prepare_time_series_graph_data(df,
                                                       x_col=x_col,
                                                       y_col=y_col,
                                                       x_range=x_range,
                                                       x_bucket=x_bucket,
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
                            yrange='[0:*]',
                            term="pdf size %dcm, %dcm" % GRAPH_PLOT_SIZE_CM,
                            out='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, file_base))
        if x_range is not None:
            g.set(xrange='[%d:%d]' % x_range)
        g.plot_data(plot_df, *plot_cmds)

        # Save plot data
        plot_df.to_csv(os.path.join(out_dir, DATA_DIR, file_base + '.csv'))
        with open(os.path.join(out_dir, DATA_DIR, file_base + '.gnuplot'), 'w+') as f:
            f.write("\n".join(plot_cmds))


def plot_time_series_matrix(df: pd.DataFrame, out_dir: str, analysis_name: str, file_cols: List[str],
                            data_cols: List[str], matrix_x_cols: List[str], matrix_y_cols: List[str], x_col: str,
                            y_col: str, x_range: Optional[Tuple[int, int]], x_bucket: Optional[float], y_div: float,
                            x_label: str, y_label: str, point_type_indices: List[int], line_color_indices: List[int],
                            format_data_title: Callable[[DataTuple], str],
                            format_subplot_title: Callable[[any, any], str],
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
    :param matrix_x_cols: Graphs are horizontally arranged based on values of these columns
    :param matrix_y_cols: Graphs are vertically arranged based on values of these columns
    :param x_col: Name of the column that has the data for the x-axis
    :param y_col: Name of the column that has the data for the y-axis
    :param x_range: (min, max) tuple for filtering the values for the x-axis
    :param x_bucket: Size of the bucket to use for aggregating data on the x-axis
    :param y_div: Number to divide the values on the y-axis by before plotting
    :param x_label: Label for the x-axis of the generated graphs
    :param y_label: LAbel for the y-axis of the generated graphs
    :param point_type_indices: Indices of file_cols used to determine point type
    :param line_color_indices: Indices of file_cols used to determine line color
    :param format_data_title: Function to format the title of a data line, receives a data_tuple (data_cols values)
    :param format_subplot_title: Function to format the title of a subplot, receives a tuple with the values of matrix_x_cols and matrix_y_cols
    :param format_file_title: Function to format the title of a graph, receives a file_tuple (file_cols values)
    :param format_file_base: Function to format the base name of a graph file, receives a file_tuple (file_cols values)
    :param sort_matrix_x: Function to sort values of the matrix_x_cols, graphs will be arranged accordingly
    :param sort_matrix_y: Function to sort values of the matrix_y_cols, graphs will be arranged accordingly
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
        print_file_tuple = sprint_tuple(file_cols, file_tuple)
        logger.debug("Generating %s matrix %s", analysis_name, print_file_tuple)

        mx_unique = list(sort_matrix_x(unique_cross_product(df, *matrix_x_cols)))
        my_unique = list(sort_matrix_y(unique_cross_product(df, *matrix_y_cols)))
        mx_cnt = float(max(1, len(mx_unique)))
        my_cnt = float(max(1, len(my_unique)))
        sub_size = "%f, %f" % ((1.0 - MATRIX_KEY_SIZE) / mx_cnt, 1.0 / my_cnt)

        subfigures = []
        key_data = set()

        # Generate subfigures
        y_max = np.ceil(df[y_col].quantile(.99) / y_div)
        for matrix_y_idx, matrix_y_tuple in enumerate(my_unique):
            for matrix_x_idx, matrix_x_tuple in enumerate(mx_unique):
                print_subplot_tuple = sprint_tuple([*file_cols, *matrix_x_cols, *matrix_y_cols],
                                                   (*file_tuple, *matrix_x_tuple, *matrix_y_tuple))
                prepared_data = prepare_time_series_graph_data(df,
                                                               x_col=x_col,
                                                               y_col=y_col,
                                                               x_range=x_range,
                                                               x_bucket=x_bucket,
                                                               y_div=y_div,
                                                               extra_title_col=extra_title_col,
                                                               file_cols=[*file_cols, *matrix_x_cols, *matrix_y_cols],
                                                               file_tuple=(
                                                                   *file_tuple, *matrix_x_tuple, *matrix_y_tuple),
                                                               data_cols=data_cols,
                                                               point_map=point_map,
                                                               line_map=line_map,
                                                               point_type_indices=point_type_indices,
                                                               line_color_indices=line_color_indices,
                                                               format_data_title=format_data_title)
                if prepared_data is None:
                    logger.debug("No data for %s %s", analysis_name, print_subplot_tuple)
                    continue

                plot_df, plot_cmds, data_tuples = prepared_data

                # Add data for key
                key_data.update(data_tuples)

                subfigures.append(gnuplot.make_plot_data(
                    plot_df,
                    *plot_cmds,
                    title='"%s"' % format_subplot_title(*matrix_x_tuple, *matrix_y_tuple),
                    key='off',
                    xlabel='"%s"' % x_label,
                    ylabel='"%s"' % y_label,
                    xrange=None if x_range is None else ('[%d:%d]' % x_range),
                    yrange='[0:%d]' % y_max,
                    pointsize='0.5',
                    size=sub_size,
                    origin="%f, %f" % (matrix_x_idx * (1.0 - MATRIX_KEY_SIZE) / mx_cnt, matrix_y_idx / my_cnt)
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
            for data_tuple in sorted(key_data)
        ]
        subfigures.append(gnuplot.make_plot(
            *key_cmds,
            key='on inside center center vertical Right samplen 2',
            pointsize='0.5',
            size="%f, 1" % MATRIX_KEY_SIZE,
            origin="%f, 0" % (1.0 - MATRIX_KEY_SIZE),
            title=None,
            xtics=None,
            ytics=None,
            xlabel=None,
            ylabel=None,
            xrange='[0:1]',
            yrange='[0:1]',
            border=None,
        ))

        gnuplot.multiplot(
            *subfigures,
            title='"%s"' % format_file_title(*file_tuple),
            term='pdf size %dcm, %dcm' %
                 (GRAPH_PLOT_SIZE_CM[0] * MATRIX_SIZE_SKEW * mx_cnt, GRAPH_PLOT_SIZE_CM[1] * my_cnt),
            output='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, format_file_base(*file_tuple)),
        )


def plot_timing(df: pd.DataFrame, out_dir: str, analysis_name: str, file_cols: List[str],
                section_cols: List[str], tick_cols: List[str], skew_cols: List[str], y_col: str,
                x_label: str, y_label: str,
                format_file_title: Callable[[FileTuple], str],
                format_file_base: Callable[[FileTuple], str],
                format_section_title: Callable[[Tuple[any, ...]], str],
                format_tick_title: Callable[[Tuple[any, ...]], str],
                format_skew_title: Callable[[Tuple[any, ...]], str],
                sort_section: Callable[[List[Tuple[any, ...]]], List[Tuple[any, ...]]] = lambda x: sorted(x),
                sort_tick: Callable[[List[Tuple[any, ...]]], List[Tuple[any, ...]]] = lambda x: sorted(x),
                sort_skew: Callable[[List[Tuple[any, ...]]], List[Tuple[any, ...]]] = lambda x: sorted(x),
                percentile_low: int = 5, percentile_high: int = 95) -> None:
    """
    Plot a timing graph that compares the average time and min and max percentiles of a time value in multiple
    scenarios. The timing values are separated into (1) sections with a large gap in between, (2) ticks on the x-axis
    and (3) skew within a tick.

    :param df: The dataframe to read the data from
    :param out_dir: Directory where all the output files are placed
    :param analysis_name: A name for the analysis used in log statements
    :param file_cols: Column names that define values for which separate graphs are generated
    :param section_cols: Column names that define values by which the data is separated into sections
    :param tick_cols: Column names that define values by which the data is separated into ticks
    :param skew_cols: Column names that define values by which the data is skewed within a tick
    :param y_col: Column name of the column that has the data being plotted on the y-axis
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param format_file_title: Function to format the title of a graph, receives a file_tuple (file_cols values)
    :param format_file_base Function to format the base name of a graph file, receives a file_tuple (file_cols values)
    :param format_section_title Function to format the label of a section, receives a section_tuple (section_cols values)
    :param format_tick_title Function to format the label of a tick, receives a tick_tuple (tick_cols values)
    :param format_skew_title Function to format the title of a data line, receives a skew_tuple (skew_cols values)
    :param sort_section  Sort section_tuples (section_cols values) to display them in this order
    :param sort_tick Sort tick_tuples (tick_cols values) to display them in this order
    :param sort_skew Sort skew_tuples (skew_cols values) to display them in this order
    :param percentile_low Percentile to display on the lower bound [0;100]
    :param percentile_high Percentile to display on the upper bound [0;100]
    """

    assert 0 <= percentile_low <= 100
    assert 0 <= percentile_high <= 100

    def p_low(x):
        return np.percentile(x, q=percentile_low)

    def p_high(x):
        return np.percentile(x, q=percentile_high)

    create_output_dirs(out_dir)

    # Ensures same point types and line colors across all graphs
    point_map: PointMap = {}
    line_map: LineMap = {}

    # Generate graphs
    for file_tuple in unique_cross_product(df, *file_cols):
        print_file_tuple = ', '.join(["%s=%s" % (col, str(val)) for col, val in zip(file_cols, file_tuple)])
        logger.debug("Generating %s timing %s", analysis_name, print_file_tuple)

        # Filter data relevant for graph
        gdf = filter_graph_data(df, "", None, file_cols, file_tuple)
        if gdf is None:
            logger.debug("No data for %s timing %s", analysis_name, print_file_tuple)
            continue

        gdf = gdf[[*section_cols, *tick_cols, *skew_cols, y_col]]
        gdf = gdf.groupby([*section_cols, *tick_cols, *skew_cols]).aggregate(
            mean=pd.NamedAgg(y_col, np.mean),
            p_low=pd.NamedAgg(y_col, p_low),
            p_high=pd.NamedAgg(y_col, p_high)
        )

        # Make sure that all combinations of sections, ticks and skews exists (needed for gnuplot commands)
        # Generate a df with all combinations and NaN values, then update with real values keeping
        # NaN's where there are no data in gdf
        full_idx = pd.MultiIndex.from_product(gdf.index.levels)
        full_gdf = pd.DataFrame(index=full_idx, columns=gdf.columns)
        full_gdf.update(gdf)

        # Move index back to columns
        full_gdf.reset_index(inplace=True)

        # Generate indexes used to calculate x coordinate in plot
        sections_sorted = sort_section(list(unique_cross_product(full_gdf, *section_cols)))
        full_gdf['section_idx'] = full_gdf[section_cols].apply(lambda x: sections_sorted.index(tuple(x)), axis=1)
        ticks_sorted = sort_tick(list(unique_cross_product(full_gdf, *tick_cols)))
        full_gdf['tick_idx'] = full_gdf[tick_cols].apply(lambda x: ticks_sorted.index(tuple(x)), axis=1)
        skews_sorted = sort_skew(list(unique_cross_product(full_gdf, *skew_cols)))
        full_gdf['skew_idx'] = full_gdf[skew_cols].apply(lambda x: skews_sorted.index(tuple(x)), axis=1)

        full_gdf = full_gdf[['section_idx', 'tick_idx', 'skew_idx', 'mean', 'p_low', 'p_high',
                             *section_cols, *tick_cols, *skew_cols]]
        full_gdf.sort_values(by=['section_idx', 'tick_idx', 'skew_idx'], inplace=True, ignore_index=True)

        # Create graph
        section_cnt = len(sections_sorted)
        tick_cnt = len(ticks_sorted)
        skew_cnt = len(skews_sorted)

        if section_cnt * tick_cnt * skew_cnt == 0:
            logger.debug("No data for %s timing %s", analysis_name, print_file_tuple)
            continue

        x_max = (tick_cnt + 1) * section_cnt
        y_max = max(full_gdf['mean'].max(), full_gdf['p_low'].max(), full_gdf['p_high'].max())
        y_max_base = 10 ** np.floor(np.log10(y_max))
        y_max = int(max(1, np.ceil(y_max / y_max_base) * y_max_base))

        file_base = format_file_base(*file_tuple)

        g = gnuplot.Gnuplot(log=True,
                            title='"%s"' % (format_file_title(*file_tuple)),
                            key='top left samplen 2',
                            xlabel='"%s"' % x_label,
                            ylabel='"%s"' % y_label,
                            xrange="[0:%d]" % x_max,
                            yrange="[0:%d]" % y_max,
                            term="pdf size %dcm, %dcm" % VALUE_PLOT_SIZE_CM,
                            out='"%s.pdf"' % os.path.join(out_dir, GRAPH_DIR, file_base),
                            pointsize='0.5')

        # Add labels for sections
        for section_idx, section_tuple in enumerate(sections_sorted):
            g.set(label='"%s" at %f,%f center' % (
                format_section_title(*section_tuple),
                (section_idx + 0.5) * (tick_cnt + 1),
                y_max * 0.075
            ))

        # Add xtics for ticks
        g.set(xtics='out rotate (%s)' % ", ".join([
            '"%s" %d' % (format_tick_title(*tick_tuple), section_idx * (tick_cnt + 1) + tick_idx + 1)
            for tick_idx, tick_tuple in enumerate(ticks_sorted)
            for section_idx in range(section_cnt)
        ]))

        plot_cmds = [
            # using: select values for error bars (x:y:y_low:y_high)
            "every %d::%d using ($2*%d+$3+1+%f):5:6:7 with errorbars pointtype %d linecolor '%s' title '%s'" %
            (
                skew_cnt,  # point increment
                skew_idx + 1,  # start point
                tick_cnt + 1,  # sat offset
                (skew_idx + 1) * (0.8 / (skew_cnt + 1)) - 0.4,  # skew within [-0.4; +0.4]
                get_point_type(point_map, None),
                get_line_color(line_map, skew_tuple),
                format_skew_title(*skew_tuple)
            )
            for skew_idx, skew_tuple in enumerate(skews_sorted)
        ]

        g.plot_data(full_gdf, *plot_cmds)

        # Save plot data
        full_gdf.to_csv(os.path.join(out_dir, DATA_DIR, file_base + '.csv'))
        with open(os.path.join(out_dir, DATA_DIR, file_base + '.gnuplot'), 'w+') as f:
            f.write("\n".join(plot_cmds))


def analyze_netem_goodput(df: pd.DataFrame, out_dir: str, extra_title_col: Optional[str] = None):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='NETEM_GOODPUT_%gS' % x_bucket,
                         file_cols=['sat', 'rate', 'queue'],
                         data_cols=['protocol', 'pep', 'loss'],
                         x_col='second',
                         y_col='bps',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=x_bucket,
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
                         "goodput_%gs_%s_r%s_q%d" % (x_bucket, sat, rate, queue),
                         extra_title_col=extra_title_col)


def analyze_netem_goodput_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='NETEM_GOODPUT_%gS' % x_bucket,
                                file_cols=['queue'],
                                data_cols=['protocol', 'pep', 'loss'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['rate'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Goodput (kbps)",
                                point_type_indices=[2],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, loss:
                                "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                                format_subplot_title=lambda sat, rate:
                                "Goodput Evolution - %s - %.0f Mbit/s" % (sat, rate),
                                format_file_title=lambda queue: "Goodput Evolution - BDP*%d" % queue,
                                format_file_base=lambda queue: "matrix_goodput_%gs_q%d" % (x_bucket, queue),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_cwnd_evo(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='NETEM_CWND_EVO_%gS' % x_bucket,
                         file_cols=['sat', 'rate', 'queue'],
                         data_cols=['protocol', 'pep', 'loss'],
                         x_col='second',
                         y_col='cwnd',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=x_bucket,
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
                         "cwnd_evo_%gs_%s_r%s_q%d" % (x_bucket, sat, rate, queue))


def analyze_netem_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='NETEM_CWND_EVO_%gS' % x_bucket,
                                file_cols=['queue'],
                                data_cols=['protocol', 'pep', 'loss'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['rate'],
                                x_col='second',
                                y_col='cwnd',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Congestion window (KB)",
                                point_type_indices=[2],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, loss:
                                "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                                format_subplot_title=lambda sat, rate:
                                "Goodput Evolution - %s - %.0f Mbit/s" % (sat, rate),
                                format_file_title=lambda queue: "Congestion Window Evolution - BDP*%d" % queue,
                                format_file_base=lambda queue: "matrix_cwnd_evo_%gs_q%d" % (GRAPH_X_BUCKET, queue),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_packet_loss(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='NETEM_PACKET_LOSS_%gS' % x_bucket,
                         file_cols=['sat', 'rate', 'queue'],
                         data_cols=['protocol', 'pep', 'loss'],
                         x_col='second',
                         y_col='packets_lost',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=GRAPH_X_BUCKET,
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
                         "packet_loss_%gs_%s_r%s_q%d" % (x_bucket, sat, rate, queue))


def analyze_netem_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='NETEM_PACKET_LOSS_%gS' % x_bucket,
                                file_cols=['queue'],
                                data_cols=['protocol', 'pep', 'loss'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['rate'],
                                x_col='second',
                                y_col='packets_lost',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1,
                                x_label="Time (s)",
                                y_label="Packets lost",
                                point_type_indices=[2],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, loss:
                                "%s%s l=%.2f%%" % (protocol.upper(), " (PEP)" if pep else "", loss * 100),
                                format_subplot_title=lambda sat, rate:
                                "Goodput Evolution - %s - %.0f Mbit/s" % (sat, rate),
                                format_file_title=lambda queue: "Packet Loss - BDP*%d" % queue,
                                format_file_base=lambda queue: "matrix_packet_loss_%gs_q%d" % (x_bucket, queue),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_netem_rtt(df: pd.DataFrame, out_dir: str):
    df['second'] = df['seq'] / 100.0
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='NETEM_RTT_%gS' % x_bucket,
                         file_cols=['sat', 'rate', 'queue'],
                         data_cols=['loss'],
                         x_col='second',
                         y_col='rtt',
                         x_range=(0, GRAPH_PLOT_RTT_SECONDS),
                         x_bucket=x_bucket,
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
                         "rtt_%gs_%s_r%s_q%d" % (x_bucket, sat, rate, queue))


def analyze_netem_ttfb(df: pd.DataFrame, out_dir: str):
    plot_timing(df, out_dir,
                analysis_name='NETEM_TTFB',
                file_cols=['protocol', 'pep', 'queue'],
                section_cols=['sat'],
                tick_cols=['rate'],
                skew_cols=['loss'],
                y_col='ttfb',
                x_label="Satellite type, link capacity (Mbit/s)",
                y_label="Time (ms)",
                format_file_title=lambda protocol, pep, queue:
                "Time to First Byte - %s%s - BDP*%d" % (protocol.upper(), " (PEP)" if pep else "", queue),
                format_file_base=lambda protocol, pep, queue:
                "ttfb_%s%s_q%d" % (protocol, "_pep" if pep else "", queue),
                format_section_title=lambda sat: "%s" % sat.upper(),
                format_tick_title=lambda rate: "%d" % rate,
                format_skew_title=lambda loss: "%.2f%%" % (loss * 100),
                sort_section=lambda section_tuples: sorted(section_tuples, key=sat_tuple_key),
                sort_tick=lambda tick_tuples: sorted(tick_tuples),
                sort_skew=lambda skew_tuples: sorted(skew_tuples),
                percentile_low=5,
                percentile_high=95)


def analyze_netem_conn_est(df: pd.DataFrame, out_dir: str):
    plot_timing(df, out_dir,
                analysis_name='NETEM_CONN_EST',
                file_cols=['protocol', 'pep', 'queue'],
                section_cols=['sat'],
                tick_cols=['rate'],
                skew_cols=['loss'],
                y_col='con_est',
                x_label="Satellite type, link capacity (Mbit/s)",
                y_label="Time (ms)",
                format_file_title=lambda protocol, pep, queue:
                "Connection Establishment - %s%s - BDP*%d" % (protocol.upper(), " (PEP)" if pep else "", queue),
                format_file_base=lambda protocol, pep, queue:
                "conn_est_%s%s_q%d" % (protocol, "_pep" if pep else "", queue),
                format_section_title=lambda sat: "%s" % sat.upper(),
                format_tick_title=lambda rate: "%d" % rate,
                format_skew_title=lambda loss: "%.2f%%" % (loss * 100),
                sort_section=lambda section_tuples: sorted(section_tuples, key=sat_tuple_key),
                sort_tick=lambda tick_tuples: sorted(tick_tuples),
                sort_skew=lambda skew_tuples: sorted(skew_tuples),
                percentile_low=5,
                percentile_high=95)


def analyze_opensand_goodput(df: pd.DataFrame, out_dir: str, extra_title_col: Optional[str] = None):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='OPENSAND_GOODPUT_%gS' % x_bucket,
                         file_cols=['sat', 'attenuation', 'ccs', 'tbs', 'qbs', 'ubs'],
                         data_cols=['protocol', 'pep'],
                         x_col='second',
                         y_col='bps',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=x_bucket,
                         y_div=1000,
                         x_label="Time (s)",
                         y_label="Goodput (kbps)",
                         point_type_indices=[],
                         line_color_indices=[0, 1],
                         format_data_title=lambda protocol, pep:
                         "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                         format_file_title=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "Goodput Evolution - %s - %ddB - CC:%s - tbs=%s - qbs=%s - ubs=%s" %
                         (sat, attenuation, ccs, tbs, qbs, ubs),
                         format_file_base=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "goodput_%gs_%s_a%d_%s_t%s_q%s_u%s" % (x_bucket, sat, attenuation, ccs, tbs, qbs, ubs),
                         extra_title_col=extra_title_col)


def analyze_opensand_goodput_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_GOODPUT_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['ccs'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Goodput (kbps)",
                                point_type_indices=[2, 3, 4],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                                "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda sat, ccs:
                                "Goodput Evolution - %s - CC:%s" % (sat, ccs),
                                format_file_title=lambda attenuation:
                                "Goodput Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_goodput_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_goodput_cc_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_GOODPUT_CC_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['ccs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Goodput (kbps)",
                                point_type_indices=[],
                                line_color_indices=[0],
                                format_data_title=lambda ccs:
                                "CC:%s" % ccs,
                                format_subplot_title=lambda sat, protocol, pep, tbs, qbs, ubs:
                                "Goodput Evolution - %s - %s%s" % (sat, protocol.upper(), " (PEP)" if pep else ""),
                                format_file_title=lambda attenuation:
                                "Goodput Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_goodput_cc_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_goodput_bs_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_GOODPUT_BS_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['tbs', 'qbs', 'ubs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['protocol', 'pep'],
                                x_col='second',
                                y_col='bps',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Goodput (kbps)",
                                point_type_indices=[],
                                line_color_indices=[0],
                                format_data_title=lambda tbs, qbs, ubs:
                                "tbs: %s, qbs: %s, ubs: %s" % (tbs, qbs, ubs),
                                format_subplot_title=lambda sat, protocol, pep:
                                "Goodput Evolution - %s - %s%s" % (sat, protocol.upper(), " (PEP)" if pep else ""),
                                format_file_title=lambda attenuation:
                                "Goodput Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_goodput_bs_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_cwnd_evo(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='OPENSAND_CWND_EVO_%gS' % x_bucket,
                         file_cols=['sat', 'attenuation', 'ccs', 'tbs', 'qbs', 'ubs'],
                         data_cols=['protocol', 'pep'],
                         x_col='second',
                         y_col='cwnd',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=x_bucket,
                         y_div=1000,
                         x_label="Time (s)",
                         y_label="Congestion window (KB)",
                         point_type_indices=[],
                         line_color_indices=[0, 1],
                         format_data_title=lambda protocol, pep:
                         "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                         format_file_title=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "Congestion Window Evolution - %s - %ddB - CC:%s - tbs=%s - qbs=%s - ubs=%s"
                         % (sat, attenuation, ccs, tbs, qbs, ubs),
                         format_file_base=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "cwnd_evo_%gs_%s_a%d_%s_t%s_q%s_u%s" % (x_bucket, sat, attenuation, ccs, tbs, qbs, ubs))


def analyze_opensand_cwnd_evo_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_CWND_EVO_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['ccs'],
                                x_col='second',
                                y_col='cwnd',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Congestion window (KB)",
                                point_type_indices=[2, 3, 4],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                                "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda sat, ccs:
                                "Goodput Evolution - %s - CC:%s" % (sat, ccs),
                                format_file_title=lambda attenuation:
                                "Congestion Window Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_cwnd_evo_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_cwnd_evo_cc_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_CWND_EVO_CC_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['ccs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['protocol', 'pep'],
                                x_col='second',
                                y_col='cwnd',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Congestion window (KB)",
                                point_type_indices=[],
                                line_color_indices=[0],
                                format_data_title=lambda ccs:
                                "CC: %s" % ccs,
                                format_subplot_title=lambda sat, protocol, pep:
                                "Goodput Evolution - %s - %s%s" % (sat, protocol.upper(), " (PEP)" if pep else ""),
                                format_file_title=lambda attenuation:
                                "Congestion Window Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_cwnd_evo_cc_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_cwnd_evo_bs_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_CWND_EVO_BS_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['tbs', 'qbs', 'ubs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['protocol', 'pep'],
                                x_col='second',
                                y_col='cwnd',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Congestion window (KB)",
                                point_type_indices=[],
                                line_color_indices=[0],
                                format_data_title=lambda tbs, qbs, ubs:
                                "tbs: %s, qbs: %s, ubs: %s" % (tbs, qbs, ubs),
                                format_subplot_title=lambda sat, protocol, pep:
                                "Goodput Evolution - %s - %s%s" % (sat, protocol.upper(), " (PEP)" if pep else ""),
                                format_file_title=lambda attenuation:
                                "Congestion Window Evolution - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_cwnd_evo_bs_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_packet_loss(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='OPENSAND_PACKET_LOSS_%gS',
                         file_cols=['sat', 'attenuation', 'ccs', 'tbs', 'qbs', 'ubs'],
                         data_cols=['protocol', 'pep'],
                         x_col='second',
                         y_col='packets_lost',
                         x_range=(0, GRAPH_PLOT_SECONDS),
                         x_bucket=x_bucket,
                         y_div=1,
                         x_label="Time (s)",
                         y_label="Packets lost",
                         point_type_indices=[],
                         line_color_indices=[0, 1],
                         format_data_title=lambda protocol, pep:
                         "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                         format_file_title=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "Packet Loss - %s - %ddB - CC:%s - tbs=%s - qbs=%s - ubs=%s" % (
                             sat, attenuation, ccs, tbs, qbs, ubs),
                         format_file_base=lambda sat, attenuation, ccs, tbs, qbs, ubs:
                         "packet_loss_%gs_%s_a%d_%s_t%s_q%s_u%s" %
                         (x_bucket, sat, attenuation, ccs, tbs, qbs, ubs))


def analyze_opensand_packet_loss_matrix(df: pd.DataFrame, out_dir: str):
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series_matrix(df, out_dir,
                                analysis_name='OPENSAND_PACKET_LOSS_%gS' % x_bucket,
                                file_cols=['attenuation'],
                                data_cols=['protocol', 'pep', 'tbs', 'qbs', 'ubs'],
                                matrix_x_cols=['sat'],
                                matrix_y_cols=['ccs'],
                                x_col='second',
                                y_col='packets_lost',
                                x_range=(0, GRAPH_PLOT_SECONDS),
                                x_bucket=x_bucket,
                                y_div=1000,
                                x_label="Time (s)",
                                y_label="Packets lost",
                                point_type_indices=[2, 3, 4],
                                line_color_indices=[0, 1],
                                format_data_title=lambda protocol, pep, tbs, qbs, ubs:
                                "%s%s" % (protocol.upper(), " (PEP)" if pep else ""),
                                format_subplot_title=lambda sat, ccs:
                                "Goodput Evolution - %s - CC:%s" % (sat, ccs),
                                format_file_title=lambda attenuation:
                                "Packet Loss - %ddB" % attenuation,
                                format_file_base=lambda attenuation:
                                "matrix_packet_loss_%gs_a%d" % (x_bucket, attenuation),
                                sort_matrix_x=lambda xvals: sorted(xvals, key=lambda xt: sat_key(xt[0])),
                                sort_matrix_y=lambda yvals: sorted(yvals, reverse=True))


def analyze_opensand_rtt(df: pd.DataFrame, out_dir: str):
    df['second'] = df['seq'] / 100.0
    for x_bucket in {GRAPH_X_BUCKET, 1}:
        plot_time_series(df, out_dir,
                         analysis_name='OPENSAND_RTT_%gS' % x_bucket,
                         file_cols=['sat', 'attenuation'],
                         data_cols=['ccs'],
                         x_col='second',
                         y_col='rtt',
                         x_range=(0, GRAPH_PLOT_RTT_SECONDS),
                         x_bucket=x_bucket,
                         y_div=1,
                         x_label="Time (s)",
                         y_label="RTT (ms)",
                         point_type_indices=[],
                         line_color_indices=[0],
                         format_data_title=lambda ccs: "RTT cc=%s" % ccs,
                         format_file_title=lambda sat, attenuation:
                         "Round Trip Time - %s - %d dB" % (sat, attenuation),
                         format_file_base=lambda sat, attenuation:
                         "rtt_%gs_%s_a%d" % (x_bucket, sat, attenuation))


def analyze_opensand_ttfb(df: pd.DataFrame, out_dir: str):
    plot_timing(df, out_dir,
                analysis_name='OPENSAND_CONN_EST',
                file_cols=['protocol', 'pep', 'attenuation'],
                section_cols=['sat'],
                tick_cols=['ccs'],
                skew_cols=['tbs', 'qbs', 'ubs'],
                y_col='ttfb',
                x_label="Satellite type, attenuation (dB)",
                y_label="Time (ms)",
                format_file_title=lambda protocol, pep, attenuation:
                "Time to First Byte - %s%s - %ddB" % (protocol.upper(), " (PEP)" if pep else "", attenuation),
                format_file_base=lambda protocol, pep, attenuation:
                "ttfb_%s%s_a%d" % (protocol, "_pep" if pep else "", attenuation),
                format_section_title=lambda sat: "%s" % sat.upper(),
                format_tick_title=lambda ccs: "%s" % ccs,
                format_skew_title=lambda tbs, qbs, ubs: "tbs=%s, qbs=%s, ubs=%s" % (tbs, qbs, ubs),
                sort_section=lambda section_tuples: sorted(section_tuples, key=sat_tuple_key),
                sort_tick=lambda tick_tuples: sorted(tick_tuples, reverse=True),
                sort_skew=lambda skew_tuples: sorted(skew_tuples),
                percentile_low=5,
                percentile_high=95)


def analyze_opensand_conn_est(df: pd.DataFrame, out_dir: str):
    plot_timing(df, out_dir,
                analysis_name='OPENSAND_CONN_EST',
                file_cols=['protocol', 'pep', 'attenuation'],
                section_cols=['sat'],
                tick_cols=['ccs'],
                skew_cols=['tbs', 'qbs', 'ubs'],
                y_col='con_est',
                x_label="Satellite type, congestion control",
                y_label="Time (ms)",
                format_file_title=lambda protocol, pep, attenuation:
                "Connection Establishment - %s%s - %ddB" % (protocol.upper(), " (PEP)" if pep else "", attenuation),
                format_file_base=lambda protocol, pep, attenuation:
                "conn_est_%s%s_a%d" % (protocol, "_pep" if pep else "", attenuation),
                format_section_title=lambda sat: "%s" % sat.upper(),
                format_tick_title=lambda ccs: "%s" % ccs,
                format_skew_title=lambda tbs, qbs, ubs: "tbs=%s, qbs=%s, ubs=%s" % (tbs, qbs, ubs),
                sort_section=lambda section_tuples: sorted(section_tuples, key=sat_tuple_key),
                sort_tick=lambda tick_tuples: sorted(tick_tuples, reverse=True),
                sort_skew=lambda skew_tuples: sorted(skew_tuples),
                percentile_low=5,
                percentile_high=95)


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


def __analyze_all_goodput(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                          time_series_cols: List[str]) -> None:
    logger.info("Analyzing goodput")

    goodput_cols = [*time_series_cols, 'bps']
    df_goodput = pd.concat([
        parsed_results['quic_client'][goodput_cols],
        parsed_results['tcp_client'][goodput_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        analyze_netem_goodput(df_goodput, out_dir)
        analyze_netem_goodput_matrix(df_goodput, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_goodput(df_goodput, out_dir)
        analyze_opensand_goodput_matrix(df_goodput, out_dir)
        analyze_opensand_goodput_cc_matrix(df_goodput, out_dir)
        analyze_opensand_goodput_bs_matrix(df_goodput, out_dir)


def __analyze_all_cwnd_evo(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                           time_series_cols: List[str]) -> None:
    logger.info("Analyzing congestion window evolution")

    cwnd_evo_cols = [*time_series_cols, 'cwnd']
    df_cwnd_evo = pd.concat([
        parsed_results['quic_server'][cwnd_evo_cols],
        parsed_results['tcp_server'][cwnd_evo_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        analyze_netem_cwnd_evo(df_cwnd_evo, out_dir)
        analyze_netem_cwnd_evo_matrix(df_cwnd_evo, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_cwnd_evo(df_cwnd_evo, out_dir)
        analyze_opensand_cwnd_evo_matrix(df_cwnd_evo, out_dir)
        analyze_opensand_cwnd_evo_cc_matrix(df_cwnd_evo, out_dir)
        analyze_opensand_cwnd_evo_bs_matrix(df_cwnd_evo, out_dir)


def __analyze_all_packet_loss(parsed_results: dict, measure_type: MeasureType, out_dir: str,
                              time_series_cols: List[str]) -> None:
    logger.info("Analyzing packet loss")

    pkt_loss_cols = [*time_series_cols, 'packets_lost']
    df_pkt_loss = pd.concat([
        parsed_results['quic_server'][pkt_loss_cols],
        parsed_results['tcp_server'][pkt_loss_cols],
    ], axis=0, ignore_index=True)

    if measure_type == MeasureType.NETEM:
        analyze_netem_packet_loss(df_pkt_loss, out_dir)
        analyze_netem_packet_loss_matrix(df_pkt_loss, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_packet_loss(df_pkt_loss, out_dir)
        analyze_opensand_packet_loss_matrix(df_pkt_loss, out_dir)


def __analyze_all_rtt(parsed_results: dict, measure_type: MeasureType, out_dir: str) -> None:
    logger.info("Analyzing round trip time")

    rtt_cols = ['sat', 'seq', 'rtt']
    if measure_type == MeasureType.NETEM:
        rtt_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == MeasureType.OPENSAND:
        rtt_cols.extend(['attenuation', 'ccs', 'tbs', 'qbs', 'ubs'])

    df_rtt = pd.DataFrame(parsed_results['ping_raw'][rtt_cols])

    if measure_type == MeasureType.NETEM:
        analyze_netem_rtt(df_rtt, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_rtt(df_rtt, out_dir)


def __analyze_all_timing(parsed_results: dict, measure_type: MeasureType, out_dir: str) -> None:
    logger.info("Analyzing time to first byte")

    df_con_times = pd.concat([
        parsed_results['quic_timing'],
        parsed_results['tcp_timing'],
    ], axis=0, ignore_index=True)
    if measure_type == MeasureType.NETEM:
        analyze_netem_ttfb(df_con_times, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_ttfb(df_con_times, out_dir)

    logger.info("Analyzing connection establishment")
    if measure_type == MeasureType.NETEM:
        analyze_netem_conn_est(df_con_times, out_dir)
    elif measure_type == MeasureType.OPENSAND:
        analyze_opensand_conn_est(df_con_times, out_dir)


def __analyze_all_stats(parsed_results: dict, measure_type: MeasureType, out_dir: str) -> None:
    logger.info("Analyzing stats")
    if measure_type == MeasureType.OPENSAND:
        df_stats = pd.DataFrame(parsed_results['stats'])
        df_runs = pd.DataFrame(parsed_results['runs'])
        df_stats.index = df_stats.index.total_seconds()
        df_runs.index = df_runs.index.total_seconds()
        analyze_stats(df_stats, df_runs, out_dir)


def analyze_all(parsed_results: dict, measure_type: MeasureType, out_dir: str, multi_process: bool = False):
    time_series_cols = ['protocol', 'pep', 'sat', 'run', 'second']
    if measure_type == MeasureType.NETEM:
        time_series_cols.extend(['rate', 'loss', 'queue'])
    elif measure_type == MeasureType.OPENSAND:
        time_series_cols.extend(['attenuation', 'ccs', 'tbs', 'qbs', 'ubs'])

    if multi_process:
        processes = [
            mp.Process(target=__analyze_all_goodput, name='goodput',
                       args=(parsed_results, measure_type, out_dir, time_series_cols)),
            mp.Process(target=__analyze_all_cwnd_evo, name='cwnd_evo',
                       args=(parsed_results, measure_type, out_dir, time_series_cols)),
            mp.Process(target=__analyze_all_packet_loss, name='packet_loss',
                       args=(parsed_results, measure_type, out_dir, time_series_cols)),
            mp.Process(target=__analyze_all_rtt, name='rtt',
                       args=(parsed_results, measure_type, out_dir)),
            mp.Process(target=__analyze_all_timing, name='timing',
                       args=(parsed_results, measure_type, out_dir)),
        ]

        for p in processes:
            p.start()

        __analyze_all_stats(parsed_results, measure_type, out_dir)

        for p in processes:
            p.join()
    else:
        __analyze_all_goodput(parsed_results, measure_type, out_dir, time_series_cols)
        __analyze_all_cwnd_evo(parsed_results, measure_type, out_dir, time_series_cols)
        __analyze_all_packet_loss(parsed_results, measure_type, out_dir, time_series_cols)
        __analyze_all_rtt(parsed_results, measure_type, out_dir)
        __analyze_all_timing(parsed_results, measure_type, out_dir)
        __analyze_all_stats(parsed_results, measure_type, out_dir)
