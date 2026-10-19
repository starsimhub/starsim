"""The time-type arithmetic table, as the code actually behaves"""
import os, warnings
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
import starsim as ss

def t(label, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            out = fn()
            s = f'{out!r}  ({type(out).__name__})'
        except Exception as E:
            s = f'{type(E).__name__}: {str(E)[:110]}'
        ws = f'  [warn: {str(w[0].message)[:80]}]' if w else ''
    print(f'{label:42s} -> {s}{ws}')

print('--- construction ---')
t('ss.dur(3)', lambda: ss.dur(3))
t("ss.dur(3, 'days')", lambda: ss.dur(3, 'days'))
t('ss.per(0.1)', lambda: ss.per(0.1))
t('ss.prob(0.3)', lambda: ss.prob(0.3))
t('ss.freq(2)', lambda: ss.freq(2))
t('ss.Rate(2)', lambda: ss.Rate(2))
t('ss.rate(2)  (deprecated)', lambda: ss.rate(2))
t('ss.time_prob(0.2)', lambda: ss.time_prob(0.2))
t('ss.rate_prob(0.2)', lambda: ss.rate_prob(0.2))
t('ss.datedur(months=1)', lambda: ss.datedur(months=1))
t('ss.months(1)', lambda: ss.months(1))
t('ss.months(1).days', lambda: ss.months(1).days)
t('ss.datedur(months=1).days', lambda: ss.datedur(months=1).days)
t('ss.year, ss.day', lambda: (ss.year, ss.day))
t('ss.years(ss.normal(3, 1))', lambda: ss.years(ss.normal(3, 1)))
t('ss.probperyear(1.5)', lambda: ss.probperyear(1.5))

print('--- arithmetic (spec 02 table) ---')
t('ss.days(3) + ss.weeks(1)', lambda: ss.days(3) + ss.weeks(1))
t('ss.days(3) + 2', lambda: ss.days(3) + 2)
t('ss.days(10) / ss.days(5)', lambda: ss.days(10) / ss.days(5))
t('ss.perday(0.1) + ss.perday(0.2)', lambda: ss.perday(0.1) + ss.perday(0.2))
t('ss.perday(0.1) + ss.peryear(36.5)', lambda: ss.perday(0.1) + ss.peryear(36.5))
t('ss.perday(0.1) - ss.peryear(36.5)', lambda: ss.perday(0.1) - ss.peryear(36.5))
t('ss.perday(0.1) * 2', lambda: ss.perday(0.1) * 2)
t('ss.perday(0.1) * ss.days(3)  [spec: count]', lambda: ss.perday(0.1) * ss.days(3))
t('ss.days(3) * ss.perday(0.1)', lambda: ss.days(3) * ss.perday(0.1))
t('ss.freqperday(0.1) * ss.days(3)', lambda: ss.freqperday(0.1) * ss.days(3))
t('ss.probperday(0.1) * ss.days(3)', lambda: ss.probperday(0.1) * ss.days(3))
t('ss.probperday(0.1) + ss.probperday(0.1)', lambda: ss.probperday(0.1) + ss.probperday(0.1))
t('ss.probperday(0.1) * 2', lambda: ss.probperday(0.1) * 2)
t('ss.probperday(0.6) * 2', lambda: ss.probperday(0.6) * 2)
t('1 / ss.days(10)  [spec: a rate]', lambda: 1 / ss.days(10))
t('2 / ss.perday(0.1)  [spec: dur]', lambda: 2 / ss.perday(0.1))
t('2 / ss.freqperday(0.1)', lambda: 2 / ss.freqperday(0.1))
t('ss.perday(0.1) + ss.freqperday(0.1)  [E101]', lambda: ss.perday(0.1) + ss.freqperday(0.1))
t('ss.perday(0.1) + ss.probperday(0.1)', lambda: ss.perday(0.1) + ss.probperday(0.1))
t('ss.perday(0.1) / ss.days(2)', lambda: ss.perday(0.1) / ss.days(2))
t('ss.freqperday(0.1) / ss.days(2)', lambda: ss.freqperday(0.1) / ss.days(2))
t('ss.perday(0.1) / ss.perday(0.2)', lambda: ss.perday(0.1) / ss.perday(0.2))
t('ss.perday(0.1) == ss.peryear(36.5)', lambda: ss.perday(0.1) == ss.peryear(36.5))
t('ss.years(1) == ss.days(365)', lambda: ss.years(1) == ss.days(365))
t('ss.days(365) == ss.years(1)', lambda: ss.days(365) == ss.years(1))
t('1 - ss.peryear(0.1)', lambda: 1 - ss.peryear(0.1))
t('float(ss.days(3))', lambda: float(ss.days(3)))
t('ss.days(3) * ss.days(3)', lambda: ss.days(3) * ss.days(3))
t('-ss.days(3)', lambda: -ss.days(3))
t('ss.days(3) - 5', lambda: ss.days(3) - 5)
t('ss.peryear(0.1).to("day")', lambda: ss.peryear(0.1).to('day'))
t('ss.peryear(0.1).to(ss.perday)', lambda: ss.peryear(0.1).to(ss.perday))
t('ss.probperyear(0.5).to("month")', lambda: ss.probperyear(0.5).to('month'))
t('ss.peryear(0.1).to_prob(ss.days(1))', lambda: ss.peryear(0.1).to_prob(ss.days(1)))
t('ss.peryear(0.1).to_events(ss.days(1))', lambda: ss.peryear(0.1).to_events(ss.days(1)))
t('ss.peryear(0.1).to_dt()', lambda: ss.peryear(0.1).to_dt())
t('np.exp(ss.perday(0.1))', lambda: np.exp(ss.perday(0.1)))
t('np.array([1,2]) * ss.perday(0.1)', lambda: np.array([1, 2]) * ss.perday(0.1))
t('ss.perday([0.1, 0.2]) * ss.days(1)', lambda: ss.perday([0.1, 0.2]) * ss.days(1))
t('ss.perday(0.1) * ss.days([1, 2])', lambda: ss.perday(0.1) * ss.days([1, 2]))

print('--- E102 WrongDimension in a rate slot ---')
try:
    d = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', rate=ss.peryear(0.1) * ss.days(3)))
    ss.Sim(d, stop=10).init()
    print('rate=peryear*days accepted silently; recovery.rate =', d.graph.transitions.recovery.rate)
except Exception as E:
    print('rate=peryear*days ->', type(E).__name__, str(E)[:300])

print('--- E101 as coded error? ---')
try:
    ss.perday(0.1) + ss.freqperday(0.1)
except Exception as E:
    print(type(E).__name__, isinstance(E, ss.CodedError), str(E)[:120])

print('--- how many time classes / names ---')
import inspect
names = [n for n in dir(ss) if n in ss.time.__all__] if hasattr(ss.time, '__all__') else []
timeobjs = [n for n in dir(ss) if (inspect.isclass(getattr(ss, n)) and issubclass(getattr(ss, n), ss.TimePar)) or isinstance(getattr(ss, n), ss.TimePar)]
print(len(timeobjs), 'TimePar classes/instances exported:', timeobjs)
print('plus date-related:', [n for n in ['date', 'DateArray', 'Timeline', 'rate', 'time_prob', 'rate_prob'] if hasattr(ss, n)])
