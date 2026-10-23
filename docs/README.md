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

3.  Build the docs with `./render` (or `./preview` to view them with live reloading). This also builds the API reference and the machine-readable API index (`api.json`, `llms.txt`, `llms-full.txt`); to build only the index, run `python make_api.py`.

4.  The built docs will be in `./_site`.

Other commands:

- `./check_notebooks.py` runs every tutorial and user guide page and reports any errors, without building the site.
- `./publish` builds the site and publishes it to GitHub Pages.
- `./clean_all` deletes the build files and outputs.
