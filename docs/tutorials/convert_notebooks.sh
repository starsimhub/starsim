#!/bin/bash
# Convert Quarto notebooks to Jupyter notebooks

for f in *.qmd; do
    quarto convert "$f"
done