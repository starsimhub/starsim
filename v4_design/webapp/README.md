# Webapp

A tool for viewing `examples` models side by side with their `vignettes` translations, with diff-style markup and highlighting.

## Purpose

Reading a translation next to its original is the fastest way to see whether the lingua franca is doing its job. This webapp makes that comparison the default view: pick a model, see the original framework code on the left and the lingua franca version on the right, with the correspondence between them made visible.

## Intended behavior

- **Model picker**: browse the corpus by framework, paradigm, or feature exercised
- **Side-by-side panes**: original (`examples`) on the left, translation (`vignettes`) on the right, with syntax highlighting for both languages
- **Diff-style markup**: highlight what corresponds, what was simplified, and what gets uglier
- **Aligned scrolling**: keep semantically corresponding regions in view together, rather than aligning by line number
- **Annotations**: surface the vignette's commentary inline, anchored to the relevant region
- **Results comparison**: where reference outputs exist, show the original's and the translation's results together

Because the two sides are different languages, a plain textual diff will not be meaningful. The alignment needs to come from the vignette declaring the correspondence between its blocks and the original's — worth settling on a format for that early, since it affects how vignettes are written.

## Status

Not yet implemented. Stack and build tooling are undecided.

## Relationship to other folders

- Reads its content from `examples` and `vignettes`.
- Makes gaps in `design` visible by showing where translations lose or add meaning.
