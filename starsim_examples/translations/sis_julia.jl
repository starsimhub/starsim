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
    dur_inf_samples = lognormal_ex(Float64(dur_inf), 1.0, n_inf)
    ti_recovered[init_inds] .= dur_inf_samples

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
    recovered = dis.infected .& (dis.ti_recovered .<= dis.ti)
    dis.infected[recovered] .= false
    dis.susceptible[recovered] .= true
end

function update_immunity!(dis::SIS)
    """Wane immunity and update relative susceptibility"""
    ti = round(Int, dis.ti) + 1  # 1-based index
    has_imm = dis.immunity .> 0
    dis.immunity[has_imm] .*= (1 - dis.waning)
    dis.rel_sus[has_imm] .= max.(0, 1 .- dis.immunity[has_imm])
    dis.mean_rel_sus[ti] = mean(dis.rel_sus)
end

function update_results!(dis::SIS)
    """Count susceptible and infected"""
    ti = round(Int, dis.ti) + 1  # 1-based index
    dis.n_susceptible[ti] = sum(dis.susceptible)
    dis.n_infected[ti] = sum(dis.infected)
end

function infect!(dis::SIS, net::RandomNet)
    """Calculate transmission across network edges"""
    p1 = net.p1
    p2 = net.p2

    # Find edges where one end is infected and the other is susceptible
    p1_inf = dis.infected[p1] .& dis.susceptible[p2]
    p2_inf = dis.infected[p2] .& dis.susceptible[p1]

    # Apply beta: disease_beta * edge_beta * rel_sus of the target
    beta_p1 = dis.beta .* net.edge_beta[p1_inf] .* dis.rel_sus[p2[p1_inf]]
    beta_p2 = dis.beta .* net.edge_beta[p2_inf] .* dis.rel_sus[p1[p2_inf]]

    # Determine which transmissions actually occur
    targets_from_p1 = p2[p1_inf][rand(length(beta_p1)) .< beta_p1]
    targets_from_p2 = p1[p2_inf][rand(length(beta_p2)) .< beta_p2]

    # Combine and deduplicate
    uids = unique(vcat(targets_from_p1, targets_from_p2))

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
    dur_inf = lognormal_ex(dis.dur_inf, 1.0, length(uids))
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


# Tests
function test_inf(sim)
    n_inf = sim.disease.n_infected
    inf0 = n_inf[1]
    infm = maximum(n_inf)
    inf1 = n_inf[end]
    n_agents = sim.n_agents
    tests = [
        ("Initial infections are nonzero: $inf0", inf0 > 0.005 * n_agents),
        ("Initial infections start low: $inf0",   inf0 < 0.05 * n_agents),
        ("Infections peak high: $infm",           infm > 0.5 * n_agents),
        ("Infections stabilize: $inf1",           inf1 < infm),
    ]
    for (k, tf) in tests
        println(tf ? "✓ $k" : "× $k")
    end
    @assert all(last.(tests))
    return n_inf
end

n_inf = test_inf(sim)
