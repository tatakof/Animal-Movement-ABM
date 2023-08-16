

# - [ ] v0.3. Add sheep gregarious behavior, in a similar way as implemented in the 
# flocking [example](https://juliadynamics.github.io/Agents.jl/stable/examples/flock/) 
# but adapted for a discrete space. Sheep will prefer regions or groups of patches that
# already contain sheep. Sheep will contain a field `visual_distance` which will 
# define the distance to  which it can sense the surrounding sheep. 


## Load packages
using DrWatson
@quickactivate "Animal-Movement-ABM"
using Agents, Random, Distributions
using GLMakie, InteractiveDynamics

include(srcdir("agent_actions.jl"))
include(srcdir("plotting.jl"))
include(srcdir("model_actions.jl"))


## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    movement_cost::Float64
    visual_distance::Float64
end


## Model function
function initialize_model(;
    n_sheep = 40, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    sheep_reproduce = 0.001, 
    movement_cost = 1, 
    visual_distance = 5, 
    seed = 321, 

)

    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = true, metric = :chebyshev)

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
        add_agent!(
            Sheep, 
            model, 
            energy, 
            sheep_reproduce, 
            Δenergy_sheep,
            movement_cost,  
            visual_distance
        )
    end

    ### Add grass
    for p in positions(model) # This could be abstracted into a function
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1 
        model.countdown[p...] = countdown 
        model.fully_grown[p...] = fully_grown
    end 

    return model
end




"""
The agent_step function will alternate between a "normal" random walk
    and a gregarious random walk. 
"""

# Agent stepping function
agent_step! = herbivore_dynamics(; 
    walk_type = RANDOM_WALK_GREGARIOUS, 
    eat = true, 
    reproduce = true, 
    #prob_random_walk = 0.1
)


# Model stepping function
model_step! = grass_growth_dynamics()







## Initialize model
model = initialize_model()



## Visualize

fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig


abmvideo(
    "Discrete_v0.3.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    # plotkwargs..., 
)







