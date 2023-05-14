


# - [ ] v0.3. Add sheep gregarious behavior, in a similar way as implemented in the 
# flocking [example](https://juliadynamics.github.io/Agents.jl/stable/examples/flock/) 
# but adapted for a discrete space. Sheep will prefer regions or groups of patches that
# already contain sheep. Sheep will contain a field `visual_distance` which will 
# define the distance to  which it can sense the surrounding sheep. 


## Load packages
using Agents, Random, Distributions
using GLMakie, InteractiveDynamics

include("../src/agent_actions.jl")
include("../src/plotting.jl")

## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    Δenergy::Float64
    visual_distance::Float64
end

## Model function
function initialize_model(;
    n_sheep = 40, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    visual_distance = 5, 
    seed = 321, 

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
        add_agent!(
            Sheep, 
            model, 
            energy, 
            Δenergy_sheep, 
            visual_distance
        )
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


"""
The agent_step function will alternate between a "normal" random walk
    and a gregarious random walk. 
"""

function agent_step!(sheep, model, prob_random_walk = 0.3)
    if rand(model.rng, Uniform(0, 1)) < prob_random_walk
    # "Normal" random walk
        randomwalk!(sheep, model)
        sheep.energy -= 1
        eat!(sheep, model)
    else
    # Random walk towards attractor
        random_walk_gregarious(sheep, model)
        sheep.energy -= 1
        eat!(sheep, model)
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
    "test.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    plotkwargs..., 
)







