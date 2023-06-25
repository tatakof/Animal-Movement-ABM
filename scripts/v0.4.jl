
"
v0.4. Implement directed movement with pathfinding. Now sheep will alternate 
between a random walk, a directed movement towards a random point in the 
`GridSpace` and a resting behaviour. There will be a `counter` parameter that 
dictates the amount of time-steps that each agent spends in each behaviour. 
When a behaviour ends, there's a transition to another behaviour. 
The probabilities to transition to another behaviour or to stay in the 
same behaviour are uniform. This implementation will follow the 
Mixed-Agent Ecosystem Pathfinding 
[example](https://juliadynamics.github.io/Agents.jl/stable/examples/rabbit_fox_hawk/)

"




using DrWatson
@quickactivate "Animal-Movement-ABM"

using Agents, Random
using GLMakie, InteractiveDynamics
using Agents.Pathfinding
using Distributions



## Load functions
include(srcdir("agent_actions.jl"))
include(srcdir("plotting.jl"))
include(srcdir("model_actions.jl"))



@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    movement_cost::Float64
    visual_distance::Float64
end



## Model function
function initialize_model(;
    n_sheep = 20, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    sheep_reproduce = 0.004, 
    movement_cost = 1, 
    visual_distance = 5, 
    seed = 321, 
    counter = 50, 

)

    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = true, metric = :chebyshev)

    ### Model properties
    properties = (
        fully_grown = falses(griddims), 
        countdown = zeros(Int, griddims), 
        regrowth_time = regrowth_time, 
        pathfinder = AStar(space), 
        # behav = Array{Symbol, 1}, 
        behav = zeros(Int, 1), 
        behav_counter = zeros(Int, 1),
        counter = counter,  
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


    
    ### Set behaviour counter. 
    model.behav_counter[1] = counter


    return model
end




## Define agent_step!
agent_step! = make_agent_stepping(; 
    walk_type = ALTERNATED_WALK,
    eat = true, 
    reproduce = true
)

# Model stepping function
model_step! = make_model_stepping()


## Initialize model
model = initialize_model()


## Visualize
fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig




abmvideo(
    "Discrete_v0.4.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    # plotkwargs..., 
)
