import sciris as sc
import numpy as np
from starsim.time import Dur, YearDur, DateDur, date, TimeVec, timeprob, Rate, perday   # Import the classes from Starsim so that Dur is an ss.Dur rather than just a bare Dur etc.
import starsim as ss

def loc(module, sim, uids):
    return np.array([Dur(x) for x in range(uids)])


module = sc.objdict(t=sc.objdict(dt=Dur(days=1)))
d = ss.normal(loc, Dur(days=1), module=module, strict=False)
d.init()
d.rvs(10)

date(1500)
date(1500.1)
YearDur(1)*np.arange(5)
np.arange(5)*YearDur(1)


t = Time(start=2001, stop=2003, dt=ss.years(0.1))

Dur(weeks=1)/Dur(days=1)

ss.perweek(1)+ss.perday(1)


ss.peryear(10)
# module = sc.objdict(t=sc.objdict(dt=Dur(weeks=1)))
# d = ss.poisson(ss.peryear(10), module=module, strict=False)
# d.init()
# d.rvs(5)


DateDur(weeks=1) - DateDur(days=1)

date(2020.1)

ss.date(2050) - ss.date(2020)


print(Dur(weeks=1)+Dur(1/52))
print(Dur(1/52)+Dur(weeks=1))
print(Dur(weeks=1)/Dur(1/52))
print(DateDur(YearDur(2/52)))

# date('2020-01-01') + Dur(weeks=52)  # Should give us 30th December 2020
date('2020-01-01') + 2.5*Dur(weeks=1)  # Should give us 30th December 2020
date('2020-01-01') + 52*Dur(weeks=1)  # Should give us 30th December 2020
date('2020-01-01') + 52*Dur(1/52) # Should give us 1st Jan 2021


import pickle
x = date('2020-01-01')
s = pickle.dumps(x)
pickle.loads(s)

t = TimeVec(date('2020-01-01'), date('2020-06-01'), Dur(days=1))
t = TimeVec(date('2020-01-01'), date('2020-06-01'), Dur(months=1)) # Allowed

t = TimeVec(Dur(days=0), Dur(days=30), Dur(days=1))
t = TimeVec(Dur(days=0), Dur(months=1), Dur(days=30))
t = TimeVec(Dur(days=0), Dur(years=1), Dur(weeks=1))
t = TimeVec(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed
t = TimeVec(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed


t = TimeVec(Dur(0), Dur(1), Dur(1/12)).init()
t.tvec+date(2020)

# date('2020-01-01')+50*Dur(days=1)
t = TimeVec(date('2020-01-01'), date('2030-06-01'), Dur(days=1))

t = TimeVec(date(2020), date(2030.5), Dur(0.1)).init()
t.tvec + YearDur(1)

print(1/YearDur(1))
print(2/YearDur(1))
print(4/YearDur(1))
print(4/DateDur(1))
print(0.5/DateDur(1))

print(perday(5)*Dur(days=1))


# time_prob
p = timeprob(0.1, Dur(years=1))
p*Dur(years=2)
p * Dur(0.5)
p * Dur(months=1)

p = timeprob(0.1, Dur(1))
p*Dur(years=2)
p * Dur(0.5)
p * Dur(months=1)

2/Rate(0.25)
1/(2*Rate(0.25))
Rate(0.5)/Rate(1)


# Dists
module = sc.objdict(t=sc.objdict(dt=Dur(days=1)))
d = ss.normal(Dur(days=6), Dur(days=1), module=module, strict=False)
d.init()
d.rvs(5)

module = sc.objdict(t=sc.objdict(dt=Dur(weeks=1)))
d = ss.normal(Dur(days=6), Dur(days=1), module=module, strict=False)
d.init()
d.rvs(5)

Rate(4)/Rate(3)

2*Rate(5)


ss.perday(1)+ss.perweek(1)

ss.perweek(1)+ss.perday(1)


# Another example
# 5/DateDur(weeks=1)
# Out[3]: <Rate: per days=1,hours=9,minutes=36> (approx. 259.99999999999994 per year)
# 259/365
# Out[4]: 0.7095890410958904
# 5/7*365
# Out[5]: 260.7142857142857
# r = 5/DateDur(weeks=1)
# r*DateDur(days=1)
# Out[7]: 0.7142857142857142
# r*DateDur(days=5)
# Out[8]: 3.5714285714285707
# r*DateDur(days=7)
# Out[9]: 4.999999999999999
# r*DateDur(weeks=1)