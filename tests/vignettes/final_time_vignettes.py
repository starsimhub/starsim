import numpy as np
import starsim as ss

#%% A: ss.Timeline repr

# 1. Explicit, days
t1 = ss.Timeline(ss.date('2020-01-01'), ss.date('2020-06-01'), ss.dur(days=1))
# Old <Timeline t=2020.01.01, ti=0, 2020.01.01-2020.06.01 dt=<DateDur: days=1>>
# × Timeline(2020.01.01—2020.06.01, dt=day; not initialized)
# ✓ Timeline(2020.01.01—2020.06.01, dt=day)

# 2. Explicit, months
t2 = ss.Timeline(ss.date('2020-01-01'), ss.date('2020-06-01'), ss.dur(months=1)) # Allowed
# × <Timeline t=2020.01.01, ti=0, 2020.01.01-2020.06.01 dt=<DateDur: months=1>>
# ✓ Timeline(2020.01.01—2020.06.01, dt=month)

# 3. Days
t3 = ss.Timeline(ss.dur(days=0), ss.dur(days=30), ss.dur(days=1)).init()
# × <Timeline t=<DateDur: 0>, ti=0, <DateDur: 0>-<DateDur: days=30> dt=<DateDur: days=1>>
# ✓ Timeline(0—30 days, dt=day)

# 4. Years
t4 = ss.Timeline(ss.dur(years=0), ss.dur(years=30), ss.dur(years=1)).init()
# × <Timeline t=<DateDur: 0>, ti=0, <DateDur: 0>-<DateDur: days=30> dt=<DateDur: days=1>>
# ✓ Timeline(0—30 years, dt=year)

# 5. Fractional years
t4 = ss.Timeline(ss.dur(years=0), ss.dur(years=30), ss.dur(years=0.1)).init()
# × <Timeline t=<DateDur: 0>, ti=0, <DateDur: 0>-<DateDur: days=30> dt=<DateDur: days=1>>
# ✓ Timeline(0—30 years, dt=years(0.1))


#%% B: Timeline additional API

# 1. Integer inputs
t1 = ss.Timeline(0, 30)
t2 = ss.Timeline(0, 30, 'day')
t2 = ss.Timeline(0, 30, 'days') # With or without "s"
t3 = ss.Timeline(0, 30, 'months')

# 2. Dates with durations, as strings
t4 = ss.Timeline('1990-01-01', 30) # Not allowed
t5 = ss.Timeline('1990-01-01', dur=30) # Allowed

# 3. Just dates, with singleton
t6 = ss.Timeline('2020-01-01', '2020-06-01', ss.week)
t7 = ss.Timeline('2020-01-01', dur=18, dt=ss.month) # Final date should be 2021-07-01

# 4. Just duration, starts at 0
t8 = ss.Timeline(dur=50, dt='week')


#%% C: ss.Timeline reconciliation

# 1. Modules inherit everything from the sim by default
ts1 = ss.Timeline(0, 30)
tm1 = ss.Timeline()
# ✓ tm1 = Timeline(0—30 years, dt=year)

# 2. Modules cannot have a different type of unit than the sim
ts2 = ss.Timeline(0, 30)
tm2 = ss.Timeline('1990-01-01') # Not allowed
# ✓ Exception('Cannot reconcile relative sim start (0) and absolute module start (1990-01-01')')

# 3. If the sim uses years, everything is converted to years
ts3 = ss.Timeline(2000, 2030)
tm3 = ss.Timeline('2010-01-01', '2025-01-01')
# ✓ tm.yearvec used for reconciliation

# 4. If the sim uses dates, everything is converted to dates
ts4 = ss.Timeline('2010-01-01', '2025-01-01')
tm4 = ss.Timeline(2000, 2030)
# ✓ tm.datevec used for reconciliation

# 5. Modules can't start before or after the sim
ts5 = ss.Timeline(2000, 2030)
tm5 = ss.Timeline(1990, 2040)
# ✓ Exception('Cannot run module from 1990-2040 since it is outside the sim timeline of 2000-2030')

# 6. Everything is rounded to the nearest day
ts6 = ss.Timeline(2020, 2021, dt=ss.month)
tm6 = ss.Timeline(dt='week')
# ✓ Dates don't align, but they're all integer days, *rounded* (not floored) to the nearest day

# 7. ...unless dt is <1 day
ts7 = ss.Timeline(2020, 2021, dt=ss.month)
tm7 = ss.Timeline(dt=ss.days(0.5))
# ✓ No rounding happens, and you see times


#%% D: TimePars repr and basic math

# 1. Singleton objects
ss.year
# × <YearDur: 1 years>
# ✓ year
ss.month # Month acts as a DateOffset with date math, otherwise as 1/12th of a year; ditto for week and day
# × <DateDur: months=1>
# ✓ month

# 2. Wrapping arrays
ss.years(np.arange(10))
# × <YearDur: [0 1 2 3 4] years>
# ✓ years([0, 1, 2, 3, 4])

# 3. Same types use exact arithmetic
ss.weeks(3) + ss.days(5)
# × <DateDur: weeks=3,days=5>
# ✓ days(26)
ss.years(2) + ss.months(3)
# ✓ <YearDur: 2.25 years> # Except for repr
# ✓ years(2.25)

# 4. If different base units (days and years), convert to years
ss.months(2) + ss.days(3)
# × <DateDur: months=2,days=3>
# ✓ years(0.174886)
ss.months(2) + ss.weeks(3)
# × <DateDur: months=2,weeks=3>
# ✓ years(224201)
ss.years(2) + ss.days(3)
# ✓ <YearDur: 2.0082191780821916 years> # Except for repr with sig figs!
# ✓ years(2.00822)
ss.years(2) + ss.weeks(3)
# ✓ <YearDur: 2.0575342465753423 years> # Ditto
# ✓ years(2.05753)


#%% E: TimePar conversions

# 1. TimePar objects can be interconverted
ss.years(2).to('days')
ss.years(2).to(ss.day)
ss.years(2).to(ss.days)
ss.years(2).to(ss.days()) # Probably no one will use, but equivalent to ss.day
ss.days(ss.years(2))
# ✓ All: days(730)
ss.days(730).to('years')
# ✓ years(2)
ss.week.to('days')
# ✓ days(7)
ss.weeks(2).to('days')
# ✓ days(14)
ss.weeks(2).to('months')
# ✓ months(0.460274)
ss.peryear(2).to('day')
# ✓ perday(0.0.00547945)
ss.peryear(2).to(ss.perday)
# ✓ perday(0.0.00547945)
ss.peryear(2).to(ss.days)
# ✓ perday(0.0.00547945)

# 2. Can mix rates/timeprobs and durs, but not vice versa
ss.peryear(2).to(ss.days)
# ✓ perday(0.0.00547945)
ss.years(2).to(ss.perday)
# ✓ Exception('Cannot convert a duration to a timeprob')
ss.peryear(2).to(ss.rateperyear)
# ✓ rateperday(2)

# 2. TimePar objects can be mutated to a different class
dur = ss.years(2).mutate('days')
# ✓ dur = days(730)
rate = ss.rateperday(2).mutate('years')
# ✓ rate = rateperday(730)


#%% F: All shortcuts
# All are classes

# 1. Duration shortcuts
ss.day
ss.week
ss.month
ss.year
ss.days()
ss.weeks()
ss.months()
ss.years()

# 2. TimeProb shortcuts
ss.perday()
ss.perweek()
ss.permonth()
ss.peryear()

# 3. Rate shortcuts
ss.rateperday()
ss.rateperweek()
ss.ratepermonth()
ss.rateperyear()

# 4. InstProb shortcuts
ss.iprobperday()
ss.iprobperweek()
ss.iprobpermonth()
ss.iprobperyear()
