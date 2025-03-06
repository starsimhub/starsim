# Starsim docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the Starsim docs at http://docs.starsim.org.

To build the docs, follow these steps:

1. Install Quarto: https://quarto.org/docs/get-started/

2.  Make sure the Python dependencies are installed:
    ```
    pip install -r requirements.txt
    ```

3.  Build the documents with `./build_docs` (requires Starsim to be installed as well).

4.  The built documents will be in `./_build`.