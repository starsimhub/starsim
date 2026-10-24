```mermaid
---
title: sir (stratified by age: 0-14, 15-64, 65+)
---
flowchart LR
    S(["S"])
    I(["I"])
    R(["R"])
    D(["D"])
    S -.->|"infection: beta * I"| I
    I -->|"recovery: dur = dur (p = 0.95, rest)"| R
    I -->|"recovery: dur = dur (p = 0.95, rest)"| D
    null1(( )) -->|"births: peryear(0.02) * (S + I + R + D).n"| S
```
