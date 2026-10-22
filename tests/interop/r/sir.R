# Build and run a declared model from R, as in "Starsim 4 in ten minutes" (docs/ten_minutes.qmd): reticulate calls Python with no Python frame above it
library(reticulate)
ss <- import("starsim")

st <- ss$states("S I R") # Named up front: there is no Python variable to name them from
S <- st[[1]]; I <- st[[2]]; R <- st[[3]]
sir <- ss$Disease(
    infection = ss$Transition(S, I, rate = 0.3 * I),
    recovery  = ss$Transition(I, R, dur = 10),
    init_prev = 0.01,
    name = "sir"
)
sim <- ss$Sim(sir, stop = 100)
sim$run()
n_I <- py_to_r(sim$results$sir$n_I$values)
cat("Peak infected:", max(n_I), "\n")

msim <- sim$new(solver = "binomial")$run(n_runs = 20L) # A count is an integer: 20L, not 20

# Checks (not in the docs)
stopifnot(max(n_I) > n_I[1]) # The epidemic grows
runs <- py_to_r(msim$results$sir$n_I$values)
stopifnot(nrow(runs) == 20, max(runs) > runs[1, 1]) # 20 runs, which grow too
