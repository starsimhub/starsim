'''
Simple tests
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
import pylab as pl
import matplotlib.pyplot as plt

#%% Define the tests

def test_microsim():

    sc.heading('Minimal sim test')
    pars = dict(
        start=1980,
        end=2020,
        location='nigeria',
    )
    sim = ss.Sim(pars)
    sim.run()
    return sim


def test_networks():
    sc.heading('Testing out different network structures')

    # Create a simulation with demographics from a given location
    people_objs = []
    labels = []
    for i, (geostruct, geomixing) in enumerate(zip([10, 10, 1], [np.repeat(0.1, 9),np.repeat(1, 9), [1]])):

        pars = dict(
            start=1980,
            end=2020,
            location='nigeria',
            geostructure=geostruct,
            geo_mixing_steps=geomixing,
        )

        sim = ss.Sim(pars=pars)
        sim.run()
        people_objs.append(sim.people)
        label = f'{geostruct} clusters, {geomixing[0] if geomixing is not None else geomixing} mixing'
        labels.append(label)

    fig, axes = pl.subplots(nrows=len(labels), ncols=3, figsize=(14, 10), sharey='col')
    for i, people in enumerate(people_objs):
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        rships = np.zeros((3, len(people.age_bin_edges)))
        overall_rships = np.zeros((3,len(people.age_bin_edges)))
        for lk, lkey in enumerate(['m', 'c', 'o']):
            active_ages = people.age[(people.n_rships[lk,:] >= 1)]
            n_rships_active = people.n_rships[:,(people.n_rships[lk,:] >= 1)]
            age_bins_active = np.digitize(active_ages, bins=people.age_bin_edges) - 1

            all_ages = people.age
            n_rships_all = people.n_rships
            age_bins_all = np.digitize(all_ages, bins=people.age_bin_edges) - 1

            for ab in np.unique(age_bins_active):
                inds = age_bins_active==ab
                rships[lk,ab] = n_rships_active[lk,inds].sum()/len(ss.true(inds))

            for ab in np.unique(age_bins_all):
                inds = age_bins_all==ab
                overall_rships[lk,ab] = n_rships_all[lk,inds].sum()/len(ss.true(inds))

            ax = axes[i, lk]
            yy = rships[lk,:]
            ax.bar(people.age_bin_edges, yy, width=2, label='|>=1 partner')
            yy = overall_rships[lk,:]
            ax.bar(people.age_bin_edges+2, yy, width=2, color='orange', label='Overall')
            ax.set_xlabel(f'Age')
            ax.set_title(f'Number of relationships, {lkey}')
        axes[i, 0].set_ylabel(labels[i])

    axes[0, 0].legend()
    fig.tight_layout()
    fig.show()

    return sim

def test_network_sex():
    sc.heading('Testing out network output by sex')

    # Create a simulation with demographics from a given location
    geostruct = 1
    pars = dict(
        start=1980,
        end=2020,
        location='nigeria',
        geostructure=geostruct
    )

    sim = ss.Sim(pars=pars)
    sim.run()
    label = f'{geostruct} clusters'
    people = sim.people
    fig, axes = pl.subplots(nrows=1, ncols=3, figsize=(14, 10))
    font_size = 15
    font_family = 'Libertinus Sans'
    pl.rcParams['font.size'] = font_size
    pl.rcParams['font.family'] = font_family

    rships_female = np.zeros((3, len(people.age_bin_edges)))
    rships_male = np.zeros((3, len(people.age_bin_edges)))
    for lk, lkey in enumerate(['m', 'c', 'o']):
        active_ages_male = people.age[(people.n_rships[lk, :] >= 1) * people.is_male]
        active_ages_female = people.age[(people.n_rships[lk, :] >= 1) * people.is_female]
        n_rships_male = people.n_rships[:, (people.n_rships[lk, :] >= 1) * people.is_male]
        n_rships_female = people.n_rships[:, (people.n_rships[lk, :] >= 1)*people.is_female]
        age_bins_male = np.digitize(active_ages_male, bins=people.age_bin_edges) - 1
        age_bins_female = np.digitize(active_ages_female, bins=people.age_bin_edges) - 1

        for ab in np.unique(age_bins_male):
            inds = age_bins_male == ab
            rships_male[lk, ab] = n_rships_male[lk, inds].sum() / len(ss.true(inds))

        for ab in np.unique(age_bins_female):
            inds = age_bins_female == ab
            rships_female[lk, ab] = n_rships_female[lk, inds].sum() / len(ss.true(inds))

        ax = axes[lk]
        yy = rships_male[lk, :]
        ax.bar(people.age_bin_edges, yy, width=2, label='Male')
        yy = rships_female[lk, :]
        ax.bar(people.age_bin_edges + 2, yy, width=2, color='orange', label='Female')
        ax.set_xlabel(f'Age')
        ax.set_title(f'Number of relationships, {lkey}')

    fig.suptitle(label)

    axes[0].legend()
    fig.tight_layout()
    fig.show()

    return sim


def test_hiv():
    sc.heading('Testing out people and pop updating with HIV')

    # Create a simulation with demographics from a given location
    geostruct = 1
    pars = dict(
        start=1995,
        end=2020,
        location='nigeria',
        geostructure=geostruct
    )

    sim = ss.Sim(pars=pars, modules=[ss.HIV])
    sim.run()

    return


def test_coinfection():
    sc.heading('Testing out coinfection with connectors')

    def gonorrhea_hiv_connector(sim):
        low_cd4_inds = ss.true(sim.people.hiv.cd4 < 200)
        sim.people.gonorrhea.rel_sus[low_cd4_inds] = 2 # increase susceptibility to NG among those with low CD4
        sim.people.gonorrhea.rel_trans[low_cd4_inds] = 2 # increase transmission of NG among those with low CD4
        return

    # Create a simulation with demographics from a given location
    pars = dict(
        start=1995,
        end=2020,
        location='nigeria',
        networks = [ss.sspop.DynamicSexualLayer()],
        connectors=[gonorrhea_hiv_connector], # this is where/how we provide connectors from HIV to other modules
    )

    sim = ss.Sim(pars=pars, modules=[ss.HIV(), ss.Gonorrhea()])
    sim.run()

    return sim


def test_layers():
    sc.heading('Testing out people, network configuration, and pop updating with HIV')

    # Create a simulation with demographics from a given location
    pars = dict(
        start=1995,
        end=2020,
        location='nigeria',
        networks = [ss.sspop.DynamicSexualLayer()],
    )

    sim = ss.Sim(pars=pars, modules=[ss.HIV()])
    sim.run()

    return sim

def test_pregnant():
    sc.heading('Testing out people, network configuration, and pop updating with HIV and pregnancy')

    def gonorrhea_hiv_connector(sim):
        low_cd4_inds = ss.true(sim.people.hiv.cd4 < 200)
        sim.people.gonorrhea.rel_sus[low_cd4_inds] = 2 # increase susceptibility to NG among those with low CD4
        sim.people.gonorrhea.rel_trans[low_cd4_inds] = 2 # increase transmission of NG among those with low CD4
        return

    # Create a simulation with demographics from a given location
    pars = dict(
        start=1995,
        end=2020,
        location='nigeria',
        networks=[ss.sspop.DynamicSexualLayer(), ss.sspop.Maternal()],
        connectors=[gonorrhea_hiv_connector],  # this is where/how we provide connectors from HIV to other modules

    )

    sim = ss.Sim(pars=pars, modules=[ss.HIV, ss.Gonorrhea, ss.Pregnancy])
    sim.run()

    plt.figure()
    plt.plot(sim.tvec, sim.results.hiv.prevalence, label='HIV')
    plt.plot(sim.tvec, sim.results.gonorrhea.prevalence, label='Gonorrhea')
    plt.title('HIV/Gonorrhea')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(sim.tvec, sim.results.pregnancy.births)
    plt.title('Births')
    plt.show()

    return



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # sim = test_microsim()
    # sim = test_networks()
    # sim = test_network_sex()
    # sim = test_hiv()

    # sim = test_layers()

    # sim = test_coinfection()
    sim = test_pregnant()

    sc.toc(T)
    print('Done.')