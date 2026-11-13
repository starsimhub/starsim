# MNCH examples

Reusable example modules for Maternal, Newborn, and Child Health (MNCH) modeling in Starsim.

## Modules

### Maternal infections (`maternal_infections.py`)

- **`CongenitalDisease`**: Simple SIR with congenital outcomes, scheduled by its own `set_congenital()` (which its `set_prognoses()` calls for each person infected before birth) and `step_congenital()`. Infected mothers transmit to unborn via `PrenatalNet`; at transmission, outcomes (stillborn, congenital infection, normal) are sampled and scheduled around delivery — stillbirths a timestep early so they occur in utero, the rest at delivery.

### Neonatal sepsis (`neonatal_sepsis.py`)

- **`NeonatalSepsis`**: SIR-like disease that infects a fraction of newborns at birth and kills some within days. Useful for testing neonatal death detection in the `Pregnancy` module.

### Fetal health (`fetal_health.py`)

- **`FetalHealth`**: Module that tracks fetal growth and birth weight outcomes (LBW, VLBW, SGA) during pregnancy; other modules change them through its methods.
- **`fetal_infection`**: Connector that links a disease (default: an SIR written as code, `ssl.AgentSIR(name='sir')`) to fetal health outcomes. Applies timing shifts (preterm birth) and growth restriction to pregnancies where the mother is infected.
- **`treat_pregnant`**: A delivery (as `ss.treat()` is, with `coverage=`, `who=`, `when=`) that treats infected pregnant women, curing infection and partially reversing the fetal damage.

## Usage

```python
import starsim as ss
import starsim.library as ssl

# Congenital disease example
sim = ss.Sim(
    diseases=ssl.mnch.CongenitalDisease(beta=0.2),
    demographics=ss.Pregnancy(),
    networks=[ss.PrenatalNet(), ss.RandomNet()],
)

# Fetal health example
sim = ss.Sim(
    diseases=ssl.AgentSIR(name='sir', beta=0.1), # Written as code, since treat_pregnant() sets its states directly
    connectors=ssl.mnch.fetal_infection(),
    interventions=ssl.mnch.treat_pregnant(disease='sir'),
    modules=ssl.mnch.FetalHealth(),
    demographics=ss.Pregnancy(),
    networks=[ss.PrenatalNet(), ss.RandomNet()],
)
```
