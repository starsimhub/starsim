import pytest
from starsim import TimeStep

def test_timestep_values():
    assert TimeStep.DAY.value == 1/365
    assert TimeStep.WEEK.value == 7/365
    assert TimeStep.MONTH.value == 30.41666666666667/365
    assert TimeStep.YEAR.value == 1

def test_timestep_describe():
    assert TimeStep.DAY.describe == ('DAY', 1/365)
    assert TimeStep.WEEK.describe == ('WEEK', 7/365)
    assert TimeStep.MONTH.describe == ('MONTH', 30.41666666666667/365)
    assert TimeStep.YEAR.describe == ('YEAR', 1)

def test_timestep_name():
    assert TimeStep.DAY.Name() == 'DAY'
    assert TimeStep.WEEK.Name() == 'WEEK'
    assert TimeStep.MONTH.Name() == 'MONTH'
    assert TimeStep.YEAR.Name() == 'YEAR'

def test_timestep_repr():
    assert repr(TimeStep.DAY) == 'DAY'
    assert repr(TimeStep.WEEK) == 'WEEK'
    assert repr(TimeStep.MONTH) == 'MONTH'
    assert repr(TimeStep.YEAR) == 'YEAR'

def test_timestep_float():
    assert float(TimeStep.DAY) == 1/365
    assert float(TimeStep.WEEK) == 7/365
    assert float(TimeStep.MONTH) == 30.41666666666667/365
    assert float(TimeStep.YEAR) == 1.0

def test_timestep_str():
    assert str(TimeStep.DAY) == str(1/365)
    assert str(TimeStep.WEEK) == str(7/365)
    assert str(TimeStep.MONTH) == str(30.41666666666667/365)
    assert str(TimeStep.YEAR) == str(1.0)
    
    
if __name__ == '__main__':
    pytest.main()

    