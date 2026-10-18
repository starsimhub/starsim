# Declare a model in R with rstarsim's transition() and disease() (its R/declare.R, found by test_r.py), and read the results as a data frame
library(reticulate)
source(file.path(Sys.getenv("STARSIM_RSTARSIM"), "R", "declare.R"))
ss <- import("starsim")

seir <- disease(
  transition("S -> E", beta = ~ beta, source = "I"),
  transition("E -> I", dur = ~ latent),
  transition("I -> R | D", rate = ~ gamma, p = c(0.99, 0.01)),
  beta = 0.3, latent = 5, gamma = 0.1, init = list(I = 10), name = "seir"
)
sim <- ss$Sim(seir, stop = 120L, verbose = 0L)
invisible(sim$run())
df <- results_df(sim)
cat("Peak infected:", max(df$seir_n_i), "\n")

# The same rate written with a state in it gives the same epidemic
sir <- function(...) disease(..., transition("I -> R", rate = ~ gamma), gamma = 0.1, init = list(I = 10), name = "sir")
peak <- function(d) { s <- ss$Sim(d, stop = 100L, verbose = 0L); invisible(s$run()); max(results_df(s)$sir_n_i) }
a <- peak(sir(transition("S -> I", beta = ~ beta), beta = 0.3))
b <- peak(sir(transition("S -> I", rate = ~ beta * I$frac), beta = 0.3))

# Checks
stopifnot(max(df$seir_n_i) > 10, df$seir_n_d[nrow(df)] > 0) # It spreads, and some die
stopifnot(abs(a - b) < 1e-6*a) # Transmission and its formula agree
