# Build and run a declared model from R through reticulate, which calls Python with no Python frame above it, so the disease is named from its states ("sir")
library(reticulate)
ss <- import("starsim")

st <- ss$states("S I R") # Named up front: there is no Python variable to name them from
S <- st[[1]]; I <- st[[2]]; R <- st[[3]]
sir <- ss$Disease(
    infection = ss$Transition(S, I, rate = 0.3 * I),
    recovery  = ss$Transition(I, R, dur = 10),
    init_prev = 0.01
)
sim <- ss$Sim(sir, stop = 100)
invisible(sim$run())
n_I <- py_to_r(sim$results$sir$I$values)
cat("Peak infected:", max(n_I), "\n")

msim <- sim$run(solver = "binomial", n_runs = 20) # Built again for the solver, with no sim$new()

# Checks
stopifnot(max(n_I) > n_I[1]) # The epidemic grows
runs <- py_to_r(msim$results$sir$I$values)
stopifnot(nrow(runs) == 20, max(runs) > runs[1, 1]) # 20 runs, which grow too
