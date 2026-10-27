"""
Test that the catalog of errors in the design spec (v4_design/design/15-errors.md) is the one the code makes
"""

# %% Imports and settings
import sciris as sc
import starsim as ss

design = sc.path(__file__).parent.parent.parent / 'v4_design' / 'design'


def test_error_catalog():
    """ The catalog of errors, warnings, and notes in 15-errors.md is the one the code makes """
    assert ss.errors.check_catalog(design / '15-errors.md'), "15-errors.md does not match the code: run ss.errors.check_catalog('v4_design/design/15-errors.md', write=True)"
    for cls in ss.errors.classes():
        assert getattr(ss.errors, cls.__name__) is cls and cls.__doc__, f'{cls.__name__} should be ss.errors.{cls.__name__}, with a docstring saying when it is raised'
    return


if __name__ == '__main__':
    test_error_catalog()
