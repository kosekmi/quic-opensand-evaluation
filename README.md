# Parse & Analyze

These scripts are used to parse the data of a measurement into pandas dataframes
and analyze these data afterwards by generating plots with gnuplot.

## Installation

1. Copy the scripts to a folder of your choice (make sure they are in the same folder)
2. Install the dependencies in `requirements.txt`

## Usage

Execute the `parse.py` script and specify input and output folder (can also be the same)
```bash
parse.py -i <input_dir> -o <output_dir>
```

### Arguments

- `-i <input_dir>` - Specify the directory to read the measurement data from
- `-o <output_dir>` - Specify the directory to store the parse & analyze results in
- `-p` - Parse only mode, parse the data and save the resulting dataframes, but don't analyze them
- `-a` - Analyze only mode, read the saved dataframes from the input directory and analyze the data
