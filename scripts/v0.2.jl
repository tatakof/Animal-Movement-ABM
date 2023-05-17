
# - [ ] v0.2. Add a home-range feature to the `agent_step!` function in order to 
# avoid that sheep end up diffusing through space. This feature will consist in 
# defining a point in space that will be the `attractor`, which will be the 
# point where the sheep gravitate towards.  




## Load packages
using DrWatson
@quickactivate "Animal-Movement-ABM"
using Agents, Random, Distributions
using GLMakie, InteractiveDynamics
using LinearAlgebra




include(srcdir("agent_actions.jl"))
include(srcdir("plotting.jl"))
include(srcdir("model_actions.jl"))

## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64 #
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
    sheep_reproduce = 0.004, 
    movement_cost = 1, 
    visual_distance = 5, 
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
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep, movement_cost, visual_distance)
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




# The agent_step function will alternate between a "normal" random walk
# and a weighted random walk to the attractor. 

agent_step! = make_agent_stepping(; walk_type = RANDOM_WALK_ATTRACTOR, eat = true, reproduce = true)




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







