import sciris as sc
import numpy as np
from starsim.time import Dur, YearDur, DateDur, date, Time, TimeProb   # Import the classes from Starsim so that Dur is an ss.Dur rather than just a bare Dur etc.
import starsim as ss

def loc(module, sim, uids):
    return np.array([Dur(x) for x in range(uids)])


module = sc.objdict(t=sc.objdict(dt=Dur(days=1)))
d = ss.normal(loc, Dur(years=1), module=module, strict=False)
d.init()
d.rvs(10)

date(1500)
date(1500.1) #CK Think this should be rounded: <1500.02.06 11:59:59>
YearDur(1)*np.arange(5) #CK Would prefer <Years=1> rather than <YearDur: 1 years>
np.arange(5)*YearDur(1) #CK Concerned that these are an array of objects

print(Dur(weeks=1)+Dur(1/52)) #CK Instead of <DateDur: weeks=2, +00:34:36.9>, I think this should be <Days=14>
print(Dur(1/52)+Dur(weeks=1)) #CK Instead of <YearDur: 0.03839572474069394 years>, I think this should be 2/52 (slightly more)
print(Dur(weeks=1)/Dur(1/52)) #CK Instead of 0.996, I think this should be 1.0
print(DateDur(YearDur(2/52))) #CK Instead of <DateDur: weeks=2, +01:09:13.8>, I think this should be <Days=14>


t = Time(date('2020-01-01'), date('2020-06-01'), Dur(days=1)) #CK Prefer TimeVecs('2020-01-01', '2020-06-01', 'day')
t = Time(date('2020-01-01'), date('2020-06-01'), Dur(months=1)) # Allowed

t = Time(Dur(days=0), Dur(days=30), Dur(days=1))
t = Time(Dur(days=0), Dur(months=1), Dur(days=30))
t = Time(Dur(days=0), Dur(years=1), Dur(weeks=1))
t = Time(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed
t = Time(Dur(days=0), Dur(years=1), Dur(months=1)) # Not allowed


t = Time(date(2020), date(2030.5), Dur(0.1)).init()
t.tvec + YearDur(1) #CK Instead of array([<2021.01.01>, <2021.02.06 11:59:59>, <2021.03.15 00:00:00>, would prefer nearest dates



# time_prob
p = TimeProb(0.1, Dur(years=1)) #CK Prefer ss.perday() and ss.rateperday()
p*Dur(years=2)
p * Dur(0.5)
p * Dur(months=1)


# Dists
module = sc.objdict(t=sc.objdict(dt=Dur(days=1)))
d = ss.normal(Dur(days=6), Dur(days=1), module=module, strict=False) #CK Prefer ss.days(ss.normal(6, 1)) or ss.normal(ss.days(6), 1)
d.init()
d.rvs(5)


# %% CK sim examples
s1 = ss.Sim(start=2020, stop=2025, dt=1/52, networks='random', diseases=ss.SIS()).run() #CK Prefer numerical years rather than 'Running 2020.03.11 09:13:50'

s2 = ss.Sim(start=2020, stop=2025, dt=ss.Dur(weeks=1), networks='random', diseases=ss.SIS()).run() #CK Prefer this to be the same as above

s3 = ss.Sim(start='2020-01-01', stop='2021-01-01', dt=ss.Dur(days=1), networks=ss.RandomNet(dt=ss.Dur(weeks=1)), diseases=ss.SIS(dt=ss.Dur(days=7))).run() # This looks right
