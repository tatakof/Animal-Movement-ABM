"Find out why are there so many sheep at the end (doesn't seem to be due to the new make_agent_stepping())" 

# v0.1. The space is `GridSpace`. Model has property `grass` (randomly distributed),
#  with a simple time-based regrowth dynamic. Agents have `energy` and 
#  a `\Deltaenergy` fields. Agents move randomly (decreasing current energy 
#  levels) and eat grass when available (increasing current energy levels). 

## Activate project
using DrWatson
@quickactivate "Animal-Movement-ABM"

## Load packages
using Agents, Random
using InteractiveDynamics
using GLMakie

## Load functions
include(srcdir("agent_actions.jl"))
include(srcdir("plotting.jl"))
include(srcdir("model_actions.jl"))



## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    movement_cost::Float64
end

## Model function
function initialize_model(;
    n_sheep = 40, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    sheep_reproduce = 0.004, 
    movement_cost = 1, 
    seed = 321

)

    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = true)

    ### Model properties
    properties = (
        fully_grown = falses(griddims), 
        countdown = zeros(Int, griddims), 
        regrowth_time = regrowth_time
    )

    
    model = AgentBasedModel(Sheep, space;
        properties, 
        rng, 
        scheduler = Schedulers.randomly 
    )

    ### Add agents
    for _ = 1:n_sheep
        energy = rand(model.rng, 1:(Δenergy_sheep*5)) - 1
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep, movement_cost)
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


# Agent stepping function

agent_step! = make_agent_stepping(;walk_type = RANDOM_WALK, eat = true, reproduce = true)



## Initialize model
model = initialize_model()



fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig


abmvideo(
    "Discrete_v0.1.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    # plotkwargs..., 
)





"