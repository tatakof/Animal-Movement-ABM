


# - [ ] v0.2. Add a home-range feature to the `agent_step!` function in order to 
# avoid that sheep end up diffusing through space. This feature will consist in 
# defining a point in space that will be the `attractor`, which will be the 
# point where the sheep gravitate towards.  



## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions"])

## Load packages
using Agents, Random, Distributions
using GLMakie, InteractiveDynamics
using LinearAlgebra

include("../src/agent_actions.jl")
include("../src/plotting.jl")

## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    Δenergy::Float64
end

## Model function
function initialize_model(;
    n_sheep = 40, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    seed = 321

)

    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = true)

    ### Model properties
    properties = (
        fully_grown = falses(griddims), 
        countdown = zeros(Int, griddims), 
        regrowth_time = regrowth_time, 
        attractor = griddims ./ 2
    )

    
    model = AgentBasedModel(Sheep, space;
        properties, 
        rng, 
        scheduler = Schedulers.randomly 
    )

    ### Add agents
    for _ = 1:n_sheep
        energy = rand(model.rng, 1:(Δenergy_sheep*5)) - 1
        add_agent!(Sheep, model, energy, Δenergy_sheep)
    end

    ### Add grass
    for p in positions(model)
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1 
        model.countdown[p...] = countdown 
        model.fully_grown[p...] = fully_grown
    end 

    return model
end


model = initialize_model()


# The agent_step function will alternate between a "normal" random walk
# and a weighted random walk to the attractor. 

function agent_step!(agent, model, prob_random_walk = 0.9)
    if rand(model.rng, Uniform(0, 1)) < prob_random_walk
    # "Normal" random walk
        randomwalk!(agent, model)
        agent.energy -= 1
        eat!(agent, model)
    else
    # Random walk towards attractor
        random_walk_to_attractor(agent, model)
        agent.energy -= 1
        eat!(agent, model)
    end
end

## Model step 

function model_step!(model)
    @inbounds for p in positions(model)
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0 
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end



## Initialize model
model = initialize_model()



## Visualize

fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig

abmvideo(
    "Discrete_v0.2.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    # plotkwargs..., 
)







