# CK docs review

Review of Codex's and Claude's pass at reworking the docs.


## Codex review

Located in `/tmp/ssv4-docs-oct24/docs/_site/index.html`

### Structure

- **Home**
- **Ten minutes**
- **Coming from**
- **What's new**
- **Tutorials**
  - Core
    - Getting started
    - Building model
    - Repeated runs
    - Age groups / contacts
    - Interventions / scenarios
    - Data / calibration
    - Places / movement
  - Advanced
    - Contact networks
    - Demographics
    - Structured compartmental
    - Importing
    - Sesonality
    - Code module
    - Disease interactions
- **User guide**
  - Writing models
    - Building / checking models
    - Diseases / transitions
    - Parameters
    - Stratification
    - Outcomes
    - Models as code
  - Populations
    - Compartments / agents
    - Demographics
    - Contacts / networks
    - Population data
    - Places / movement
  - Running
    - Creating / running sims
    - Timepars
    - Plotting
    - Repeated runs
    - Solver
    - Pop scaling
    - Scripts (?!)
  - Interventions
    - Interventions
    - Products
    - Interactions
  - Calibration
    - Calibration
    - Diagnostics
  - Interoperability
    - Interoperability
    - Library models
    - Choosing
  - Troubleshooting
    - Troubleshooting
    - Profiling
    - Results
    - Arrays
    - Random numbers
    - Distributions
    - Embed Starsim
- **Architecture**
  - Model as data
  - Model to run
  - Solvers
  - Mixing
  - Randomness
  - Model spec
  - Testing

- **Gallery**

### CK comments

- Tutorials
  - Too much duplication -- the same topic appears in multiple places in the docs, which is fine for a tutorial and user guide, but not more duplication than that
  - Don't really understand the flow
  - Not clear what's more compartmental vs agent-based -- use ∿ ∴ to distinguish
  - Gotta move "Coming from" away from the top
  - Add result.df as a shortcut for result.to_df()
  - Calibration is ... terrible in T6
  - A2 S plot looks weird since mixes infection with births
  - A, C labels etc don't make sense
  - Tutorial content is generally good -- measles example is compelling and cool!
  - Running ss.demo() both takes a long time to compile, and first run is still slow. Should run ODE anyway, with ss.demo(solver='abm') instead or something
  - Seasonality going further is too brief and random -- and all examples should compare with and without the new feature being added
  - Small code module tutorial sucks
- User guide
  - Pages are too brief, and too much overlap with tutorials
  - Minor: scrolling is hard since resets position in scrollbar every time you click, should remember
  - Why are some things in sim.pars an sc.obj, and others an ss.ndict()?
- Other
  - Too many headings, don't fit
  - Merge "In ten minutes" into the home/readme
  