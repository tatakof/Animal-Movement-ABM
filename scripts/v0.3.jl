


# - [ ] v0.3. Add sheep gregarious behavior, in a similar way as implemented in the 
# flocking [example](https://juliadynamics.github.io/Agents.jl/stable/examples/flock/) 
# but adapted for a discrete space. Sheep will prefer regions or groups of patches that
# already contain sheep. Sheep will contain a field `visual_distance` which will 
# define the distance to  which it can sense the surrounding sheep. 



## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions"])

## Load packages
using Agents, Random
using GLMakie, InteractiveDynamics
using Distributions

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



"""
    random_walk_gregarious(sheep, model)

Performs a gregarious random walk for the given `sheep` agent in the `model`. The function first determines
possible locations the sheep can move to and the neighboring agents within the sheep's visual distance. If there are
neighboring agents, the sheep moves towards the closest neighbor, considering the Euclidean distance between
them. The sheep's movement is biased using a probability distribution based on the distance to the closest
neighbor, where closer locations have higher probabilities.

If there are no neighboring agents, the sheep performs a simple random walk, selecting a neighboring position
uniformly at random.

# Arguments
- `sheep`: The Sheep agent to perform the random walk.
- `model`: The agent-based model containing the sheep and other agents.

# Examples
```julia
sheep = model[1] # Get the first sheep agent in the model
random_walk_gregarious(sheep, model) # Perform a gregarious random walk for the sheep
```
"""
function random_walk_gregarious(sheep, model)
    # Get the nearby locations that an agent can move to
    possible_locations = [pos for pos in nearby_positions(sheep.pos, model)]

    # Get the ids of the nearby agents
    neighbor_ids = [ids for ids in nearby_ids(sheep, model, sheep.visual_distance)]

    # If there are neighboring agents, do a weighted walk toward the closest
    if length(neighbor_ids) > 0
        # Find the closest neighbor
        closest_neighbor_pos = nothing
        min_distance = Inf
        for ids in neighbor_ids
            neighbor_pos = model[ids].pos
            distance = euclidean_distance(neighbor_pos, sheep.pos, model)
            if distance < min_distance
                min_distance = distance
                closest_neighbor_pos = neighbor_pos
            end
        end

        # Compute the euclidean distance of each neighboring position to the closest neighbor
        eudistance_to_closest_neighbor = [euclidean_distance(pos, closest_neighbor_pos, model) for pos in possible_locations]

        # Define function that computes the probs of moving in each direction
        f(x) = exp(-x / 3)

        # Compute the probs of moving in each direction according to the distance to the attractor
        probs_to_move = f.(eudistance_to_closest_neighbor) ./ sum(f.(eudistance_to_closest_neighbor))

        # now we sample the movements using the probs_to_move
        move_to = wsample(1:length(eudistance_to_closest_neighbor), probs_to_move)

        # and move towards that location
        move_agent!(sheep, possible_locations[move_to], model)
    else
        # Do a random walk
        move_agent!(sheep, possible_locations[rand(model.rng, 1:8)], model)
    end
end











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

offset(a) = (-0.1, -0.1*rand(model.rng)) 
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
    # heatarray = homecolor, 
    # heatkwargs = heatkwargs, 
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







