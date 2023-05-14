using DrWatson

@quickactivate "Animal-Movement-ABM"

# v0.1. The space is `GridSpace`. Model has property `grass` (randomly distributed),
#  with a simple time-based regrowth dynamic. Agents have `energy` and 
#  a `\Deltaenergy` fields. Agents move randomly (decreasing current energy 
#  levels) and eat grass when available (increasing current energy levels). 


## Install packages
# using Pkg
# Pkg.add(["Agents", "Tables", "Random", "GLMakie", "InteractiveDynamics", "CairoMakie"])

## Load packages
using Agents, Random
using InteractiveDynamics

## Load functions
include(srcdir("agent_actions.jl"))
include(srcdir("plotting.jl"))

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
    move_cost = 1, 
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
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep, move_cost)
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

initialize_model()

# Agent stepping function
function agent_step!(agent, model)
    randomwalk!(agent, model)
    agent.energy -= agent.movement_cost
    if agent.energy < 0 
        kill_agent!(agent, model)
        return 
    end
    eat!(agent, model; food = :grass)
    if rand(model.rng) ≤ agent.reproduction_prob
        reproduce!(agent, model)
    end
end





function make_agent_stepping(; eat = false, go_home = true)
    custom_agent_step! = function(agent, model)  #modified name of anonymous `agent_step!` function to avoid name collision
        print("hello")
    end
end

test = make_agent_stepping()
test()


function make_agent_stepping(; eat = false, go_home = true) # avoided anonymous function definition because it raises an error
    custom_agent_step! = function(agent, model) 
        if go_home
            dir = random_direction_towards_home(agent, home, model)
            walk!(agent, dir, model)
        end
        if eat
            eat!(agent, model, :grass)
        end
    end
    return custom_agent_step!
end

custom_agent_step! = make_agent_stepping()

help(custom_agent_step!)


println(methods(custom_agent_step!))

using InteractiveUtils

println(@code_lowered custom_agent_step!(agent, model))

step_function1 = make_agent_stepping()
println(@code_lowered step_function1(agent, model))



## Initialize model
model = initialize_model()

using CairoMakie
using GLMakie

#

fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig


abmvideo(
    "test.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    # plotkwargs..., 
)





