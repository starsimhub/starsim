# MNCH Examples

Reusable example modules for Maternal, Newborn, and Child Health (MNCH) modeling in Starsim.

## Modules

### Maternal infections (`maternal_infections.py`)

- **`CongenitalDisease`**: Simple SIR with congenital outcomes via the generic `set_congenital`/`step_congenital` framework in the base `Infection` class. Infected mothers transmit to unborn via `PrenatalNet`; at transmission, outcomes (stillborn, congenital infection, normal) are sampled and scheduled for delivery.

### Neonatal sepsis (`neonatal_sepsis.py`)

- **`NeonatalSepsis`**: SIR-like disease that infects a fraction of newborns at birth and kills some within days. Useful for testing neonatal death detection in the `Pregnancy` module.

### Fetal health (`fetal_health.py`)

- **`fetal_infection`**: Connector that links a disease (default: SIR) to fetal health outcomes. Applies timing shifts (preterm birth) and growth restriction to pregnancies where the mother is infected. Also reverses damage when treatment occurs.
- **`treat_pregnant`**: Intervention that treats infected pregnant women each timestep, curing infection and enabling `fetal_infection` to reverse damage.

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
    diseases=ss.SIR(beta=0.1),
    connectors=ssl.mnch.fetal_infection(),
    interventions=ssl.mnch.treat_pregnant(disease='sir'),
    custom=ssl.mnch.FetalHealth(),
    demographics=ss.Pregnancy(),
    networks=[ss.PrenatalNet(), ss.RandomNet()],
)
```
