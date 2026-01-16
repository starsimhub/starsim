import starsim as ss
import numpy as np
import pytest

@pytest.mark.parametrize('age_range,expected', [
    ('5-9', (5, 9)),
    ('5 to 9', (5, 9)),
    ('<5', (0, 5)),
    ('95+', (95, np.inf)),
    ('>95', (95, np.inf)),
])
def test_parse_age_range_valid(age_range, expected):
    assert ss.parse_age_range(age_range) == expected

@pytest.mark.parametrize('age_range', [
    '-1-4',
    '-5',
])
def test_parse_age_range_invalid(age_range):
    with pytest.raises(Exception):
        ss.parse_age_range(age_range)
