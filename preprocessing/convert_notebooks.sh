#!/bin/bash
# helper script to convert all jupyter notebooks in the preprocessing folder to python scripts without having to open up jupyter lab

# Get the directory of the script (preprocessing folder)
NOTEBOOK_DIR="$(dirname "$0")"

echo "Starting conversion of Jupyter notebooks to Python scripts in directory: $NOTEBOOK_DIR"

# Convert all .ipynb files in the directory to .py files
for notebook in "$NOTEBOOK_DIR"/*.ipynb; do
    # Extract the filename without the path
    filename=$(basename -- "$notebook")
    
    echo "Converting $filename to Python script..."
    
    # Run the conversion
    jupytext --to py "$notebook"
    
    # Check if the conversion was successful
    if [ $? -eq 0 ]; then
        echo "Successfully converted $filename to ${filename%.ipynb}.py"
    else
        echo "Failed to convert $filename"
    fi
done

echo "Conversion process completed."

