# Fixtures defined in this file will be used by all tests

import starsim as ss
import pytest

# Run all tests with both single and multi RNG streams
# @pytest.fixture(params=['single','multi'], autouse=True)
# def set_rng_type(request):
#     original_rng_type = ss.options.multirng
#     if request.param == 'single':
#         ss.options.multirng = False
#     else:
#         ss.options.multirng = True
#     yield
#     ss.options.multirng = original_rng_type