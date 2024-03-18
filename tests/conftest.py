# Fixtures defined in this file will be used by all tests

import starsim as ss
import pytest

# Run all tests with both single and multi RNG streams
@pytest.fixture(params=['centralized', 'single', 'multi'], autouse=True)
def set_rng_type(request):
    original_rng_type = ss.options.rng
    ss.options.rng = request.param
    yield
    ss.options.rng = original_rng_type