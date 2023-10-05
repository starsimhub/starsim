"""
Vignette 10: Custom people
"""

import stisim as ss
import numpy as np

##################
# @RomeshA
##################
#Custom people attributes

# Make a new module

# Derive from people
class MyPeople(ss.People):
    def __init__(self):
        super().__init__()
        self.states['risk_tolerance'] = State('risk_tolerance', float, False)
        self.something_else = 'bar'

    def update_pre(self):
        super().update_pre()
        <do something related to the new attributes>
        <or could overload another function>


##################
# @pausz
##################
# The People class in STIsim is a crucial component that handles various operations related to the simulated population.
# In this vignette, we show how to extend and customize the People class by adding new attributes and methods.

# Case 0: Add custom method to people
class CustomPeople(ss.People):
    """
    A custom class that extends the People class to add new functionality.

    Args:
        states (dict, optional): A dictionary of custom states to add to the people.

    **Examples**::
        ppl = CustomPeople(states=new_states)
    """

    def __init__(self, states=None, *args, **kwargs):
        """ Initialize the CustomPeople object. """
        super().__init__(*args, states=states, **kwargs)

    def plot_state(self, states, **kwargs):
        """
        Plot the distribution of specific states among the population.
        This method could plot all pairs of states as 2D scatterplots, or 2D histograms,
        to detect associations. If the state is boolean, then plot a swarmplot.
        """
        # Handle grid of subplots if multiple are passed
        self.plot_single_state()


# Case 1: Add new states to people
new_states = [ss.State('is_vaccinated', bool, False),                      # Everyone starts unvaccinated
              ss.State('ti_vaccinated', ss.default_float, np.nan),         # Track when a person is vaccinated
              ss.StochState('fertility_level', ss.default_float,           # Fertility levels
                            distdict=dict(dist='normal',
                                          par1=[0.5],
                                          par2=[0.1])),
              ss.State('infertile', bool, False),                          # Add a new base state that tracks who is fertile
              ss.State('ti_infertile', ss.default_float, np.nan),
              ss.StochState('has_access_to_medical_care', bool,            # Add a new base state
                            distdict=dict(dist='choice',
                                          par1=[1, 0],
                                          par2=[0.45, 0.55]))
]



# Create the people with new states
ppl = ss.CustomPeople(states=new_states)

# Once we have customised the People class by adding new base states,
# we can utilise them in the simulation.
sim = ss.Sim(people=ppl,
             network_structure='random',
             modules=ss.gonorrhea())

# Accessing and utilising custom attributes
sim.ppl.plot_state(['fertilty_rate', 'has_access_to_medical_care')


##################
# @robynstuart
##################
# See example in v09_custom_networks.py, i.e.
is_fsw = ss.StochState('fsw', bool, distdict=dict(dist='choice', par1=[1, 0], par2=[0.05, 0.95]))
is_client = ss.StochState('client', bool, distdict=dict(dist='choice', par1=[1, 0], par2=[0.15, 0.85]))
ppl = ss.People(100, states=[is_fsw, is_client])
sex_work = sex_work()  # Question: what constraints are there on this? Must it be a ss.Network or could it be anything?
syph = ss.syphilis()
sim = ss.Sim(people=ppl, networks=['mf', sex_work], modules=syph)
