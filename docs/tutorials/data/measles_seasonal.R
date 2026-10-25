## Measles over 20 years: an SEIR model in odin, with births, deaths, and seasonal transmission
## (made for the Starsim tutorial "Bringing a model in"; times are in days)

deriv(S) <- mu * N - beta * S * I / N - mu * S
deriv(E) <- beta * S * I / N - sigma * E - mu * E
deriv(I) <- sigma * E - gamma * I - mu * I
deriv(R) <- gamma * I - mu * R

N <- S + E + I + R
beta <- beta0 * (1 + amp * cos(2 * 3.141593 * t / 365)) # Highest at the start of each year

initial(S) <- 7000
initial(E) <- 30
initial(I) <- 30
initial(R) <- 92940

beta0 <- user(1.9)   # Mean transmission rate, per day
amp <- user(0.2)     # Size of the seasonal cycle, relative to the mean
sigma <- user(0.1)   # 1 / latent period (10 days)
gamma <- user(0.125) # 1 / infectious period (8 days)
mu <- user(0.00008)  # Birth and death rate, per day (about 3% a year)
