# Contributing

Welcome! We are thrilled you are interested in contributing to Starsim. This document will help you get started.

- We are serious about inclusion and believe the open-source software community still has a long way to go. The Starsim community follows a [code of conduct](https://docs.idmod.org/projects/starsim/en/stable/conduct.html). By participating in this project, you agree to abide by its terms.
- Take a look at our house [style guide](https://github.com/starsimhub/styleguide). Starsim more or less follows Google's Python style guide, but with some exceptions.
- Feel free to [open an issue](https://github.com/starsimhub/starsim/issues/new/choose) on more or less anything! This project is small enough that we don't need a formal triage system.
- Pull requests should be made against `main`. In addition to following the [style guide](https://github.com/starsimhub/styleguide), please make sure the tests pass (`run_tests` in the `tests` folder; they also run via GitHub actions).

If you have any other questions, please reach out to us: <info@starsim.org>. Thank you!

## Release procedure
This assumes you're about to release version `3.1.1`.
1. Merge all PRs into the release candidate branch (`rc3.1.1`)
2. Update the version in `version.py` from `3.1.1.dev0` to `3.1.1`, and update the date
3. Mark the PR ready
4. If tests pass, merge
5. On the `Starsim` repository page, click 'Releases' to go to the [Releases](https://github.com/starsimhub/starsim/releases) page
6. Click 'Draft a new release' on the top right
7. In the tag dropdown, enter `v3.1.1` as a new tag. Leave the target as `main`
8. Set the Release title to `v3.1.1`
9. Generate release notes automatically, and then edit further as required
10. Click 'Publish release' to deploy, which will update PyPI and build/upload the latest documentation
11. Create a new release candidate branch (e.g. `rc3.1.2`), and make a draft PR
