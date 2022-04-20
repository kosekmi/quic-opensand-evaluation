# Parse & Analyze

These scripts are used to parse the data of a measurement into pandas dataframes
and analyze these data afterwards by generating plots with gnuplot.

To run a measurement you can use these scripts: [quic-opensand-emulation](https://github.com/kosekmi/quic-opensand-emulation)

## Installation

1. Copy the scripts to a folder of your choice (make sure they are in the same folder)
2. Install the dependencies in `requirements.txt`

## Usage

Execute the `evaluate.py` script and specify input and output folder, which can also be the same:
```bash
python evaluate.py -i <input_dir> -o <output_dir>
```

The folder `sample` contains exemplary compressed raw data (`raw_data.zip`), which can be evaluated once it is uncompressed:
```bash
python evaluate.py -i sample/raw_data -o sample/parsed
```
The generated plots of the sample data are provided in `sample/graphs`.


### Arguments

- `-a, --analyze` - Analyze only mode, read the saved dataframes from the input directory and analyze the data. Output
  directory can be omitted, if it should be the same as input
- `-d, --auto-detect` - Auto detect some analysis parameterf (e.g. measurement time) from the measurement data
- `-d, --help` - Print a help message and exit
- `-i, --input=<input_dir>` - Specify the directory to read the measurement data from
- `-m, --multi-process` - Use multiple process to parse and analyze the results
- `-o, --output=<output_dir>` - Specify the directory to store the parsed & analyzed results in
- `-p, --parse` - Parse only mode, parse the data and save the resulting dataframes, but don't analyze them

## Combined Analysis

To generate graphs from multiple independent measurements, use the `combined_analyze.py` script. But first use
the `evaluate.py` script in parse only mode to parse the measurements individually, as the combined analysis relies on
the raw parsed data.
```bash
python combined_analyze.py -o <output_dir> <title1> <path1> <title2> <path2> [... <titleN> <pathN>]
```

### Arguments

- `-o <output_dir>` - Specify the directory to write the results to
- `<titleN> <pathN>` - Specify pairs of input data. The title will be prepended to the legend in the graph.
  The path specifies the output directory, where the parsed data are found.
  (At least two pairs must be given)
