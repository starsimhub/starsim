#!/bin/bash
# Convert Quarto notebooks to Jupyter notebooks
# (although note that they can be run in Jupyter directly via jupytext)

for f in *.qmd; do
    quarto convert "$f"
done