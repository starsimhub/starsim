## Measles in a town of 100,000 people: an SEIR model in odin
## (made for the Starsim tutorial "Bringing a model in"; times are in days)

deriv(S) <- -beta * S * I / N
deriv(E) <- beta * S * I / N - sigma * E
deriv(I) <- sigma * E - gamma * I
deriv(R) <- gamma * I

N <- S + E + I + R

initial(S) <- N0 - I0
initial(E) <- 0
initial(I) <- I0
initial(R) <- 0

N0 <- user(100000)   # Population size
I0 <- user(10)       # Number infectious at the start
beta <- user(1.9)    # Transmission rate, per day
sigma <- user(0.1)   # 1 / latent period (10 days)
gamma <- user(0.125) # 1 / infectious period (8 days)
