# Assets

Fonts for Starsim's plots. Starsim adds them to Matplotlib when it is imported (`ss.settings.load_fonts()`), and uses Mulish by default; Raleway and Rosario can be chosen instead, e.g. `with ss.style(font='Rosario'):` or `sim.plot(font='Raleway')`.

| Files | Font | License |
|---|---|---|
| `Mulish-*.ttf` | Mulish | SIL Open Font License (`license-mulish.txt`) |
| `Raleway-*.ttf` | Raleway | SIL Open Font License (`license-raleway.txt`) |
| `Rosario-*.ttf` | Rosario | SIL Open Font License (`license-rosario.txt`) |

To skip loading the fonts, set the environment variable `STARSIM_INSTALL_FONTS=0` before importing Starsim. If the fonts don't show up in plots, run `ss.settings.load_fonts(rebuild=True)` to rebuild Matplotlib's font cache.

`fonts_installed_successfully` is a marker written on the first import, when Matplotlib's font cache is rebuilt with these fonts; while it exists, later imports don't rebuild the cache. It is gitignored.
