# CK feedback 1

In general, update the code to make README.md and docs/tutorials/t[1,2].qmd work. Specifically (this is major points, not the exhaustive list):

## More flexible simple sim creation

This should work:

```python
import starsim as ss

S, I, R = ss.states(3)      # Create the states
infection = S >> I @ 0.3*I  # Define infection
recovery  = I >> R @ 0.1    # Define recovery
sim = ss.Sim(infection, recovery) # Create the simulation
sim.run().plot() # Run and plot the simulation
```

As should `sim = ss.Sim([infection, recovery])` and `ss.Sim(diseases=[infection, recovery])`. With a bare list of transitions (or args), make a Module to put them in. (Module and Disease should be the same for this, no?) If there's a single unnamed module like this, put things in `sim.results.S`, `sim.results.infection` etc.

## Implicit and explicit definitions

We want sim creation to be simple, but also explicit and transparent. Currently we have `infection = S >> I @ 0.3*I` and `infection = ss.Transition(S, I, rate=0.3 * S*I/N)`. I *think* these should both work, but let's check what other disease modeling frameworks do. If they all require `beta*S*I/N` to be written explicitly, I guess we should do that too.

The main thing is to make it clear how to specify the two common forms: `dI/dt = β*S*I/N` (frequency based) and `dI/dt = β*S*I` (density based). I guess `*I` or `*S*I/N` can mean frequency based, and `*S*I` can mean density based, but we should be clear and consistent.

## Default prevalence

Let init_prev=0.01 by default, so diseases can be created implicitly (as in t1_intro). Also, maybe you have this already, but ensure this works:

```py
S = ss.State('Susceptible')
I = ss.State('Infectious', init=0.01)
R = ss.State('Recovered')
N = S + I + R
```

The first declared state starts with 1.0. If the states can't add up to 1 (e.g. S=0.99, I=0.1, R=0), raise an error.

## Rename results

Instead of `n_S` etc, I think it should just be `S` -- e.g. `sim.results.sir.S`. Unless that means something else? Then we can use dot notation like flows if we want. But not sure what derived quantities would be useful ... I guess 'new' or 'diff' or 'change' or something, but maybe easier for the user to just use the flow, or `np.diff()`.

## Direct module access

Allow modules to be accessed directly from the sim, e.g. `sim.sir` as well as `sim.diseases.sir` (if two module names conflict, print a warning and don't create these short links for either of them).

## Better plotting

The plotting sucks sorry! For an implicitly defined disease like this:

```py
S, I, R = ss.states(3)      # Create the states
infection = S >> I @ 0.3*I  # Define infection
recovery  = I >> R @ 0.1    # Define recovery
sir = ss.Disease(infection, recovery)
```

Support the following:

```py
sir.plot() # Same as 'states'
sir.plot('states') # Plot all states in the same subplot
sir.plot('flows') # Plot each flow in its own subplot
sir.plot('all') # Plot all states and flows, one subplot each
sir.plot(['S', 'I']) # Plot 'S' and 'I' in the same subplot 
sir.plot('S', 'I') # Plot 'S' and 'I' in different subplots 
```

Make the styling nice, ie normal Starsim plotting, not the monstrosity of plain Matplotlib.

