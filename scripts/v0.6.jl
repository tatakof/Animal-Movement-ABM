
# - [ ] v0.6. Use the `elevation` model property as a cost metric
#  for the path finding.  


## Load packages
using DrWatson
@quickactivate "Animal-Movement-ABM"
using Agents, Random
using GLMakie, InteractiveDynamics
import ImageMagick
using FileIO: load
using Agents.Pathfinding
using Distributions



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
    seed = 321, 
    counter = 50, 

)
    heightmap_url = "https://raw.githubusercontent.com/juliadynamics/" *
    "juliadynamics/master/videos/agents/runners_heightmap.jpg"
    ## download and load the heightmap. the grayscale value is converted to `float64` and
    ## scaled from 1 to 40
    elevation = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    ## the x and y dimensions of the pathfinder are that of the heightmap
    dims = (size(elevation))
    ## the region of the map that is accessible to sheep is defined using `bitarrays`
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
    space = GridSpace(dims, periodic = true)

    ### Model properties
    properties = (
        fully_grown = falses(dims), 
        countdown = zeros(Int, dims), 
        regrowth_time = regrowth_time, 
        behav = zeros(Int, 1), 
        behav_counter = zeros(Int, 1),
        counter = counter,  
        elevation = elevation, 
        land_walkmap = land_walkmap, 
        pathfinder = AStar(
            space; 
            walkmap = land_walkmap, 
            # Use the `elevation` model property as a cost metric
            # for the path finding. 
            cost_metric = PenaltyMap(elevation, MaxDistance{2}())
        )
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

    ### Set behaviour counter. 
    model.behav_counter[1] = counter


    return model
end

model = initialize_model()


## Agent stepping function
# Make an "alternated walk" using `elevation` model property as a cost metric
agent_step! = make_agent_stepping(; walk_type = ALTERNATED_WALK, eat = true, reproduce = true)










## Model step. 
model_step! = make_model_stepping(; walkmap = true)



## Initialize model
model = initialize_model()



## Visualize
fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
fig


# test fig
static_preplot!(ax, model) = scatter!(ax, (128, 409); color = (:red, 50), marker = 'x')

fig, ax, abmobs = abmplot(
    model;
    agent_step!, 
    model_step!, 
    figurekwargs = (resolution = (700, 700)), 
    ac = :black, 
    as = 8, 
    scatterkwargs = (strokecolor = :white, strokewidth = 2), 
    heatarray = model -> penaltymap(model.pathfinder), 
    heatkwargs = (colormap  = :terrain,), 
    static_preplot!, 
)
fig






abmvideo(
    "Discrete_v0.6.mp4", 
    model, 
    agent_step!, 
    model_step!; 
    frames = 100, 
    framerate = 8, 
    plotkwargs..., 
)



# Data collection

count_grass(model) = count(model.fully_grown)
sheep(a) = a isa Sheep

model = initialize_model()

adata = [(sheep, count)]
mdata = [count_grass]



fig, abmobs = abmexploration(model;
    agent_step!, 
    model_step!, 
    params, 
    plotkwargs...,
    adata, 
    mdata 
)

fig


