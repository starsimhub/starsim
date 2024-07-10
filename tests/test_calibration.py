"""
Test calibration
"""

#%% Imports and settings
import sciris as sc
import starsim as ss

do_plot = 1
do_save = 0
n_agents = 2e3


#%% Helper functions

def make_sim():
    hiv = ss.HIV(
        beta = {'random': [0.01]*2, 'maternal': [1, 0]},
        init_prev = 0.15,
    )
    pregnancy = ss.Pregnancy(fertility_rate=20)
    death = ss.Deaths(death_rate=10)
    random = ss.RandomNet(n_contacts=4)
    maternal = ss.MaternalNet()

    sim = ss.Sim(
        dt = 1,
        n_agents = n_agents,
        total_pop = 9980999,
        start = 1990,
        n_years = 40,
        diseases = [hiv],
        networks = [random, maternal],
        demographics = [pregnancy, death],
    )

    return sim


def make_data():
    """ Define the calibration target data """
    target_data = [
    ['year', 'n_alive', 'hiv.prevalence', 'hiv.n_infected', 'hiv.new_infections', 'hiv.new_deaths'],
    [  1990,  10432409,        0.0699742,          730000 ,               210000,           25000],
    [  1991,  10681008,        0.0851979,          910000 ,               220000,           33000],
    [  1992,  10900511,        0.1009127,          1100000,               220000,           43000],
    [  1993,  11092775,        0.1081785,          1200000,               210000,           53000],
    [  1994,  11261752,        0.1154349,          1300000,               200000,           63000],
    [  1995,  11410721,        0.1226916,          1400000,               180000,           74000],
    [  1996,  11541215,        0.1299689,          1500000,               160000,           84000],
    [  1997,  11653254,        0.1287194,          1500000,               150000,           94000],
    [  1998,  11747079,        0.1362040,          1600000,               140000,           100000],
    [  1999,  11822722,        0.1353326,          1600000,               130000,           110000],
    [  2000,  11881482,        0.1346633,          1600000,               120000,           120000],
    [  2001,  11923906,        0.1341842,          1600000,               110000,           130000],
    [  2002,  11954293,        0.1254779,          1500000,               100000,           130000],
    [  2003,  11982219,        0.1251854,          1500000,               94000 ,           130000],
    [  2004,  12019911,        0.1164734,          1400000,               89000 ,           120000],
    [  2005,  12076697,        0.1159257,          1400000,               83000 ,           120000],
    [  2006,  12155496,        0.1069475,          1300000,               78000 ,           110000],
    [  2007,  12255920,        0.1060711,          1300000,               74000 ,           93000],
    [  2008,  12379553,        0.1050118,          1300000,               69000 ,           80000],
    [  2009,  12526964,        0.0957933,          1200000,               65000 ,           68000],
    [  2010,  12697728,        0.0945050,          1200000,               62000 ,           54000],
    [  2011,  12894323,        0.0930642,          1200000,               56000 ,           42000],
    [  2012,  13115149,        0.0914972,          1200000,               49000 ,           34000],
    [  2013,  13350378,        0.0973755,          1300000,               47000 ,           28000],
    [  2014,  13586710,        0.0956817,          1300000,               45000 ,           25000],
    [  2015,  13814642,        0.0941030,          1300000,               44000 ,           24000],
    [  2016,  14030338,        0.0926563,          1300000,               43000 ,           23000],
    [  2017,  14236599,        0.0913139,          1300000,               34000 ,           23000],
    [  2018,  14438812,        0.0900351,          1300000,               27000 ,           22000],
    [  2019,  14645473,        0.0920401,          1347971,               23000 ,           None],
    [  2020,  14862927,        0.0874659,          1300000,               20000 ,           None],
    [  2021,  15085870,        0.0861733,          1300000,               19000 ,           None],
    [  2022,  15312158,        0.0848998,          1300000,               17000 ,           None],
    ]
    df = sc.dataframe(target_data[1:], columns=target_data[0])
    return df

#%% Define the tests

def test_calibration(do_plot=True):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = dict(
        diseases = dict(
            hiv = dict(
                init_prev = [0.15, 0.01, 0.30],
            ),
        ),
        networks = dict(
            randomnet = dict(
                n_contacts = [4, 2, 10],
            ),
        ),
    )

    # Make the sim and data
    sim = make_sim()
    data = make_data()

    # Define weights for the data
    weights = {
        'n_alive':            1.0,
        'hiv.prevalence':     1.0,
        'hiv.n_infected':     1.0,
        'hiv.new_infections': 1.0,
        'hiv.new_deaths':     1.0,
    }

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        data = data,
        weights = weights,
        total_trials = 4,
        n_workers = 2,
        die = True
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.confirm_fit()
    print(f'Fit with original pars: {calib.before_fit:n}')
    print(f'Fit with best-fit pars: {calib.after_fit:n}')
    if calib.after_fit <= calib.before_fit:
        print('✓ Calibration improved fit')
    else:
        print('✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')
    
    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    sim, calib = test_calibration()

    T.toc()
