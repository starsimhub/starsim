# Starsim docs

## Tutorials

Please see the `tutorials` subfolder.

## User guide

Please see the `user_guide` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the Starsim docs at http://docs.starsim.org.

To build the docs, follow these steps:

1. Install Quarto: https://quarto.org/docs/get-started/

2.  Make sure the Python dependencies are installed (as well as Starsim itself):
    ```
    pip install -r requirements.txt
    ```

3.  Build the docs with `./render` (or `./preview` to view them with live reloading). Before rendering, Quarto runs `qpyd prerender` (from [quartopydoc](https://github.com/sciris/quartopydoc), configured in the `qpyd` section of `_quarto.yml`), which builds the API reference and the machine-readable API index (`api.json`, `llms.txt`, `llms-full.txt`), and writes the version from `_variables.py` to `_variables.yml`. It skips the API build if Starsim, quartodoc and `_quarto.yml` haven't changed since the last one; run `qpyd prerender --force` to rebuild it anyway, or `python make_api.py` to build only the index.

4.  The built docs will be in `./_site`.

Other commands:

- `./check_notebooks.py` (or `qpynb check tutorials user_guide`) runs every tutorial and user guide page and reports any errors, without building the site.
- `./clean_outputs.py` (or `qpyd clean`) removes the temporary files the notebooks write; this also runs after every full render.
- `./publish` builds the site and publishes it to GitHub Pages.
- `./clean_all` deletes the build files and outputs.
- `./qrender` is a faster `./render` for working on the docs: it runs notebooks and renders pages in parallel, keeps going if a notebook fails, and only re-renders pages that changed. It needs a Quarto build with the `--jobs`, `--keep-going` and `--incremental` options (set `QUARTO` to its path).
- `quarto clean` (with the same Quarto build) removes the `*_files` folders that interrupted renders leave next to the pages.
