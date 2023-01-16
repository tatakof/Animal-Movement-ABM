


# - [ ] v0.2. Add a home-range feature to the `agent_step!` function in order to 
# avoid that sheep end up diffusing through space. This feature will consist in 
# defining a point in space that will be the `attractor`, which will be the 
# point where the sheep gravitate towards.  



## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions", "Plots"])

## Load packages
using Agents, Random
using GLMakie, InteractiveDynamics
using Distributions, Plots 


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

function agent_step!(sheep, model, prob_random_walk = 0.5)
    if rand(Uniform(0, 1)) < prob_random_walk
    # "Normal" random walk
        walk!(sheep, rand, model)
        sheep.energy -= 1
        eat!(sheep, model)
    else
    # Random walk towards attractor
        random_walk_to_attractor(sheep, model)
        sheep.energy -= 1
        eat!(sheep, model)
    end
end

"
Make a random walk with an 'attractor' which the agents will gravitate towards. 
An agent in a given position has 8 possible locations to move to. We will compute the
euclidean distance between those 8 possible locations and the 'attractor', and then 
compute some weights for moving towards each of those 8 possible locations that depend
on the euclidean distance to the attractor (closer, higher weights). Once we 
have the weights, we will do a weighted sample of the possible locations to move to, 
and then make the move. 
"
function random_walk_to_attractor(sheep, model)
    # Get the nearby locations that an agent can move to 
    possible_locations = [pos for pos in nearby_positions(sheep.pos, model)] 
    # Compute the euclidean distance of each neighboring position to the attractor
    eudistance_to_attractor = [euclidean_distance(pos, model.attractor, model) for pos in nearby_positions(sheep.pos, model)]
    # Define function that computes the probs of moving in each direction
    f(x) = exp(-x / 3)
    # Compute the probs of moving in each direction according to the distance to the attractor
    probs_to_move = f.(eudistance_to_attractor) ./ sum(f.(eudistance_to_attractor))
    # now we sample the movements using the probs_to_move
    move_to = wsample(1:length(eudistance_to_attractor), probs_to_move)
    # and move towards that location
    move_agent!(sheep, possible_locations[move_to], model)
end

# Show shape of f
using Plots
x = 1:10 
y = exp.(- x / 3)
Plots.plot(x, y)


## Describe
function eat!(sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
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







