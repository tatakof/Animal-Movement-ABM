

# v0.1. The space is `GridSpace`. Model has property `grass` (randomly distributed),
#  with a simple time-based regrowth dynamic. Agents have `energy` and 
#  a `\Deltaenergy` fields. Agents move randomly (decreasing current energy 
#  levels) and eat grass when available (increasing current energy levels). 


## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics"])

## Load packages
using Agents, Random
using GLMakie, InteractiveDynamics



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


## Agent stepping function
function agent_step!(sheep, model)
    walk!(sheep, rand, model)
    sheep.energy -= sheep.movement_cost
    if sheep.energy < 0 
        kill_agent!(sheep, model)
        return 
    end
    eat!(sheep, model)
    if rand(model.rng) ≤ sheep.reproduction_prob
        reproduce!(sheep, model)
    end
end


function eat!(sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end


function reproduce!(sheep, model)
    sheep.energy /= 2
    id = nextid(model)
    offspring = Sheep(id, sheep.pos, sheep.energy, sheep.reproduction_prob, sheep.Δenergy, sheep.movement_cost)
    add_agent_pos!(offspring, model)
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

params = Dict(
    :regrowth_time => 1:1:30, 
    :move_cost => 1:1:8
)

model = initialize_model()
fig, ax, abmobs = abmplot(model;
    agent_step!, 
    model_step!, 
    params, 
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


