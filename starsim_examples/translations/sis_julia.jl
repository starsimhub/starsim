"""
Test simulation performance -- Julia version
"""

using Random
using Statistics

const F = Float32


#%% Simulation types

struct Pars
    n_agents::Int
    dur::Int
end

mutable struct RandomNet
    n_agents::Int
    n_contacts::Int
    dur::Int
    beta::Float64
    p1::Vector{Int}
    p2::Vector{Int}
    edge_beta::Vector{F}
    edge_dur::Vector{F}
end

function RandomNet(pars::Pars; n_contacts=10, dur=0, beta=1.0)
    RandomNet(pars.n_agents, n_contacts, dur, beta, Int[], Int[], F[], F[])
end

function end_pairs!(net::RandomNet)
    """Remove expired edges"""
    net.edge_dur .-= 1
    active = net.edge_dur .> 0
    net.p1 = net.p1[active]
    net.p2 = net.p2[active]
    net.edge_beta = net.edge_beta[active]
    net.edge_dur = net.edge_dur[active]
end

function get_source(inds, n_contacts)
    """Get source array from contact counts"""
    n_half_edges = sum(n_contacts)
    source = zeros(Int, n_half_edges)
    count = 1
    for (i, person_id) in enumerate(inds)
        n = n_contacts[i]
        source[count:count+n-1] .= person_id
        count += n
    end
    return source
end

function get_edges(inds, n_contacts)
    """Create random edges by shuffling source into target"""
    source = get_source(inds, n_contacts)
    target = shuffle(source)
    return source, target
end

function add_pairs!(net::RandomNet)
    """Generate new random edges"""
    inds = 1:net.n_agents
    n_conn = fill(net.n_contacts, net.n_agents)

    # Calculate how many new edges are needed
    target_edges = sum(n_conn) / 2  # Divide by 2 since edges are bidirectional
    current_edges = length(net.p1)
    needed = target_edges - current_edges

    if needed > 0
        # Scale down contacts proportionally to only create what's needed
        scale = needed / sum(n_conn)
        n_conn = round.(Int, n_conn .* scale)

        # Get the new edges
        p1, p2 = get_edges(collect(inds), n_conn)
        beta = fill(F(net.beta), length(p1))
        if net.dur == 0
            dur = zeros(F, length(p1))
        else
            dur = fill(F(net.dur), length(p1))
        end

        # Append new edges
        append!(net.p1, p1)
        append!(net.p2, p2)
        append!(net.edge_beta, beta)
        append!(net.edge_dur, dur)
    end
end

function step!(net::RandomNet)
    """Update the network"""
    end_pairs!(net)
    add_pairs!(net)
end


mutable struct SIS
    n_agents::Int
    dur::Int
    beta::Float64
    init_prev::Float64
    dur_inf::Float64
    waning::Float64
    imm_boost::Float64
    ti::Float64
    susceptible::Vector{Bool}
    infected::Vector{Bool}
    ti_recovered::Vector{F}
    immunity::Vector{F}
    rel_sus::Vector{F}
    n_susceptible::Vector{F}
    n_infected::Vector{F}
    mean_rel_sus::Vector{F}
end

function SIS(pars::Pars; beta=0.05, init_prev=0.01, dur_inf=10, waning=0.05, imm_boost=1.0)
    n = pars.n_agents

    # Create states
    susceptible = trues(n)
    infected = falses(n)
    ti_recovered = zeros(F, n)
    immunity = zeros(F, n)
    rel_sus = ones(F, n)

    # Seed initial infections
    n_inf = round(Int, init_prev * n)
    init_inds = randperm(n)[1:n_inf]
    susceptible[init_inds] .= false
    infected[init_inds] .= true

    # Create results
    n_susceptible = zeros(F, pars.dur)
    n_infected = zeros(F, pars.dur)
    mean_rel_sus = zeros(F, pars.dur)

    SIS(n, pars.dur, beta, init_prev, Float64(dur_inf), waning, imm_boost, 0.0,
        susceptible, infected, ti_recovered, immunity, rel_sus,
        n_susceptible, n_infected, mean_rel_sus)
end

function lognormal_ex(mean, std, n)
    """Create a lognormal distribution with the specified mean and exponential SD"""
    sigma = sqrt(log(1 + (std / mean)^2))
    mu = log(mean) - sigma^2 / 2
    return exp.(mu .+ sigma .* randn(n))
end

function step_state!(dis::SIS)
    """Progress infectious -> recovered"""
    @inbounds for i in 1:dis.n_agents
        if dis.infected[i] && dis.ti_recovered[i] <= dis.ti
            dis.infected[i] = false
            dis.susceptible[i] = true
        end
    end
end

function update_immunity!(dis::SIS)
    """Wane immunity and update relative susceptibility"""
    ti = round(Int, dis.ti) + 1  # 1-based index
    wane_factor = F(1 - dis.waning)
    total = F(0)
    @inbounds for i in 1:dis.n_agents
        if dis.immunity[i] > 0
            dis.immunity[i] *= wane_factor
            dis.rel_sus[i] = max(F(0), F(1) - dis.immunity[i])
        end
        total += dis.rel_sus[i]
    end
    dis.mean_rel_sus[ti] = total / dis.n_agents
end

function update_results!(dis::SIS)
    """Count susceptible and infected"""
    ti = round(Int, dis.ti) + 1  # 1-based index
    n_sus = 0
    n_inf = 0
    @inbounds for i in 1:dis.n_agents
        n_sus += dis.susceptible[i]
        n_inf += dis.infected[i]
    end
    dis.n_susceptible[ti] = n_sus
    dis.n_infected[ti] = n_inf
end

function infect!(dis::SIS, net::RandomNet)
    """Calculate transmission across network edges"""
    p1 = net.p1
    p2 = net.p2
    n_edges = length(p1)

    # Boolean flag array for deduplication
    is_target = falses(dis.n_agents)

    @inbounds for i in 1:n_edges
        a = p1[i]
        b = p2[i]
        eb = net.edge_beta[i]

        # p1 infected -> p2 susceptible
        if dis.infected[a] && dis.susceptible[b]
            beta = dis.beta * eb * dis.rel_sus[b]
            if rand() < beta
                is_target[b] = true
            end
        end

        # p2 infected -> p1 susceptible
        if dis.infected[b] && dis.susceptible[a]
            beta = dis.beta * eb * dis.rel_sus[a]
            if rand() < beta
                is_target[a] = true
            end
        end
    end

    # Collect unique targets
    uids = findall(is_target)

    if !isempty(uids)
        set_prognoses!(dis, uids)
    end
end

function set_prognoses!(dis::SIS, uids)
    """Set prognoses for newly infected agents"""
    dis.susceptible[uids] .= false
    dis.infected[uids] .= true
    dis.immunity[uids] .+= dis.imm_boost

    # Sample duration of infection and determine recovery time
    dur_inf = lognormal_ex(dis.dur_inf, dis.dur_inf, length(uids))
    dis.ti_recovered[uids] .= dis.ti .+ dur_inf
end

function step!(dis::SIS, net::RandomNet)
    """Progress the disease by one time step"""
    step_state!(dis)
    update_immunity!(dis)
    infect!(dis, net)
    update_results!(dis)
    dis.ti += 1
end


mutable struct Sim
    n_agents::Int
    dur::Int
    disease::SIS
    network::RandomNet
    verbose::Int
    ti::Int
end

function Sim(pars::Pars; network, disease, verbose=0)
    Sim(pars.n_agents, pars.dur, disease, network, verbose, 0)
end

function step!(sim::Sim)
    step!(sim.network)
    step!(sim.disease, sim.network)
end

function run!(sim::Sim)
    for t in 1:sim.dur
        step!(sim)
    end
end


#%% Run

pars = Pars(100_000, 100)

# Warmup run (JIT compilation)
sim = Sim(pars, network=RandomNet(pars), disease=SIS(pars))
t0 = time()
run!(sim)
println("Warmup (includes JIT compilation): $(round(time() - t0, digits=3))s")

# Timed run
sim = Sim(pars, network=RandomNet(pars), disease=SIS(pars))
t0 = time()
run!(sim)
elapsed = round(time() - t0, digits=3)
println("Time for SIS-Julia, n_agents=$(pars.n_agents), dur=$(pars.dur): $(elapsed)s")
