"""
Streamlit example from the Deployment section of the User Guide.

To install dependencies:
    pip install streamlit

To run:
    streamlit run streamlit.py
"""

import streamlit as st
import starsim as ss

def run_sim(beta, n_agents):
    sis = ss.SIS(beta=beta)
    sim = ss.Sim(
        n_agents = n_agents,
        diseases = sis,
        networks = 'random',
    )
    sim.run()
    sim.label = f'Beta={beta:n} • Agents={n_agents:,} • Time={sim.timer.total:0.1f} s'
    return sim

# Create the Streamlit interface
st.title('SIS Dashboard')
beta = st.slider('Transmission rate (beta)', 0.0, 1.0, 0.1)
n_agents = st.slider('Number of agents', 1_000, 100_000, 10_000)

# Run simulation and plot results
sim = run_sim(beta, n_agents)
fig = sim.diseases.sis.plot()
fig.suptitle(sim.label)
st.pyplot(fig)