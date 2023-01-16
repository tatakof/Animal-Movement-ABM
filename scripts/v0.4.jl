
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



## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions", "Plots"])


using Agents, Random
using GLMakie, InteractiveDynamics
using Agents.Pathfinding
using Distributions



@agent Sheep GridAgent{2} begin
    energy::Float64
    Δenergy::Float64
end



## Model function
function initialize_model(;
    n_sheep = 20, 
    griddims = (80, 80), 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    seed = 321, 
    counter = 50, 

)

    rng = MersenneTwister(seed)
    space = GridSpace(griddims, periodic = true)

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
        add_agent!(Sheep, model, energy, Δenergy_sheep)
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

model = initialize_model()





# Agents will alternate between a random walk, a directed movement towards a random point in the 
# `GridSpace` and a resting behaviour. There will be a `counter` parameter that 
# dictates the amount of time-steps that each agent spends in each behaviour. 
# When a behaviour ends, there's a transition to another behaviour. 
# The probabilities to transition to another behaviour or to stay in the 
# same behaviour are uniform.
"jumps from 10 to 8, thus there's 9 steps instead of 10"
function agent_step!(sheep, model)
    if model.behav_counter[1] == model.counter 
        model.behav[1] = sample(1:3)
        model.behav_counter[1] -= 1 # this may give a silent mistake
        
        # If Directed movement, plan route
        if model.behav[1] == 2
            plan_route!(
                sheep, 
                random_walkable(model, model.pathfinder), 
                model.pathfinder
            )
        end
    end

    if 0 < model.behav_counter[1] < model.counter 
        # 1 == RandomWalk
        if model.behav[1] == 1
            walk!(sheep, rand, model)
            eat!(sheep, model)
            model.behav_counter[1] -= 1
        # 2 == Directed Walk
        elseif model.behav[1] == 2
            move_along_route!(sheep, model, model.pathfinder)
            model.behav_counter[1] -= 1
        # 3 == Rest
        elseif model.behav[1] == 3
            move_agent!(sheep, sheep.pos, model)
            model.behav_counter[1] -= 1
        end
    end

    if model.behav_counter[1] == 0
        model.behav_counter[1] = model.counter 
    end

end


function eat!(sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end


## Model step. 
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

offset(a) = (-0.1, -0.1*rand()) 
ashape(a) = :circle 
acolor(a) = RGBAf(1.0, 1.0, 1.0, 0.8) 

grasscolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (
    colormap = [:white, :green], 
    colorrange = (0, 1)
)

plotkwargs = (;
    ac = acolor,
    as = 15,
    am = ashape,
    offset,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = grasscolor,
    heatkwargs = heatkwargs,
)


model = initialize_model()
fig, ax, abmobs = abmplot(model;
    agent_step!, 
    model_step!, 
    plotkwargs...
)
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
