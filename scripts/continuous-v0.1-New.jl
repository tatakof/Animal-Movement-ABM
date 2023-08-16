# v0.1. The space is a `ContinuousSpace`. Model has property `grass` (randomly distributed),
#  with a simple time-based regrowth dynamic. Agents have `energy` and 
#  a `\Deltaenergy` fields. Agents move randomly (decreasing current energy 
#  levels) and eat grass when available (increasing current energy levels). 



using DrWatson
@quickactivate "Animal-Movement-ABM"

using Agents, Random
using GLMakie, InteractiveDynamics
using Agents.Pathfinding
using Distributions, LinearAlgebra



@agent Sheep ContinuousAgent{2} begin #contains id, pos, and vel fields

end




function initialize_model(;
    n_sheep = 1, 
    dims = (50, 50),
    seed = 123, 

)
    space2d = ContinuousSpace(dims;)
    rng = Random.MersenneTwister(seed)

    model = ABM(Sheep, space2d, scheduler = Schedulers.Randomly())

    for _ in 1:n_sheep
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(
            model, 
            vel
        )
    end
    

    return model

end


model = initialize_model(n_sheep = 1)


# https://juliadynamics.github.io/Agents.jl/stable/api/#Agents.randomwalk!
# Anything that supports rand can be used as an angle distribution instead. This can be useful to create correlated random walks.


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
    space = GridSpace(dims, periodic = true, metric = :chebyshev)

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
agent_step! = herbivore_dynamics(; walk_type = ALTERNATED_WALK, eat = true, reproduce = true)










## Model step. 
model_step! = grass_growth_dynamics(; walkmap = true)



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


function agent_step!(sheep, model)
    randomwalk!(sheep, model, )
end


using GLMakie

# const sheep_polygon = Makie.Polygon(Point2f[(-1, -1), (2, 0), (-1, 1)])

# function sheep_marker(b::Sheep)
#     φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
#     InteractiveDynamics.rotate_polygon(sheep_polygon, φ)
# end

const sheep_polygon = Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    InteractiveDynamics.scale(rotate2D(sheep_polygon, φ), 2)
end


fig, = abmplot(model; agent_step!, am = sheep_marker)


fig









# Data collection
# IMPORTANT: This function give an erroneous output when having more than 1 agent. Including an if statement to take this into account
function collect_data(adata::Vector{Symbol}; steps::Int, single_agent::Bool)
    df = run!(model, agent_step!, steps; adata)

    if single_agent
        df[1].angle_radians = get_angle.(df[1].vel)
        df[1].angle_degrees = rad2deg.(df[1].angle_radians)
    else 
        println("CAREFUL: There's more than one agent, will not compute angles")
    end
    return df
end




adata = [:pos, :vel]



df = collect_data(adata; steps = 100, single_agent = false)


using CairoMakie



function make_angle_histograms(df)
    fig = Figure()
    # Create a subplot for the angle_radians histogram
    ax1 = Axis(fig, xlabel = "Angle (Radians)", ylabel = "Frequency")
    hist!(ax1, df[1].angle_radians)
    # Create a subplot for the angle_degrees histogram
    ax2 = Axis(fig, xlabel = "Angle (Degrees)", ylabel = "Frequency")
    hist!(ax2, df[1].angle_degrees)
    # Add the subplots to the Figure
    fig[1, 1] = ax1
    fig[2, 1] = ax2
    # Show the Figure
    display(fig)
end

make_angle_histograms(df)

