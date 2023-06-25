
"
- [ ] v0.5. Add the spatial property `elevation`, which consists in a 
2D matrix, where each value denotes the height of the terrain at that point.
Also define regions where sheep cannot walk and take them into account in
the `agent_step!` function. 

"


## Load packages
using DrWatson
@quickactivate "Animal-Movement-ABM"
using Agents, Random
using GLMakie, InteractiveDynamics
import ImageMagick
using FileIO: load
using Agents.Pathfinding




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
    visual_distance::Float64
end

## Model function
function initialize_model(;
    n_sheep = 40, 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    sheep_reproduce = 0.004, 
    movement_cost = 1, 
    water_level = 1, 
    visual_distance = 5, 
    mountain_level = 18, 
    seed = 321

)
    heightmap_url = "https://raw.githubusercontent.com/JuliaDynamics/" *
                    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png"

    ## Download and load the heightmap. The grayscale value is converted to `Float64` and
    ## scaled from 1 to 40
    elevation = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    ## The x and y dimensions of the pathfinder are that of the heightmap
    dims = (size(elevation))
    ## The region of the map that is accessible to sheep is defined using `BitArrays`
    land_walkmap = BitArray(falses(dims...))
    water_map = BitArray(falses(dims...))

    # land walk map will be between `water_level` and `mountain_level`
    for i = 1:dims[1], j = 1:dims[2]
        if water_level < elevation[i, j] < mountain_level
            land_walkmap[i, j] = true
        end
    end

    # water_map will be everything less or equal to `water_level`
    for i = 1:dims[1], j = 1:dims[2]
        if elevation[i, j] ≤ water_level 
            water_map[i, j] = true
        end
    end

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = true, metric = :chebyshev)

    ### Model properties
    properties = (
        fully_grown = falses(dims), 
        countdown = zeros(Int, dims), 
        regrowth_time = regrowth_time, 
        elevation = elevation, 
        land_walkmap = land_walkmap, 
        pathfinder = AStar(space; walkmap = land_walkmap)
    )

    
    model = AgentBasedModel(Sheep, space;
        properties, 
        rng, 
        scheduler = Schedulers.randomly 
    )

    ### Add agents
    for _ = 1:n_sheep
        energy = rand(model.rng, 1:(Δenergy_sheep*5)) - 1
        add_agent_pos!(
            Sheep(
                nextid(model), 
                random_walkable(model, model.pathfinder), 
                energy, 
                sheep_reproduce, 
                Δenergy_sheep, 
                movement_cost, 
                visual_distance
            ), 
            model
        )
    end

    ### Add grass
    for p in positions(model)
        if model.land_walkmap[p...] == 1
            fully_grown = rand(model.rng, Bool)
            countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1 
            model.countdown[p...] = countdown 
            model.fully_grown[p...] = fully_grown
        end
    end 

    return model
end


## Agent stepping function
# Make a random walk taking into account the positions agents can and cannot take
agent_step! = make_agent_stepping(; walk_type = RANDOM_WALKMAP, eat = true, reproduce = true)


## Model step. 
model_step! = make_model_stepping(; walkmap = true)


## Initialize model
model = initialize_model()



## Visualize
fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig


abmvideo(
    "Discrete_v0.5.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    plotkwargs..., 
)
