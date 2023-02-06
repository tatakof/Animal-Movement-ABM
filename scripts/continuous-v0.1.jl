# v0.1. The space is a `ContinuousSpace`. Model has property `grass` (randomly distributed),
#  with a simple time-based regrowth dynamic. Agents have `energy` and 
#  a `\Deltaenergy` fields. Agents move randomly (decreasing current energy 
#  levels) and eat grass when available (increasing current energy levels). 


using Agents, LinearAlgebra
using Random
using InteractiveDynamics, CairoMakie
using FileIO: load
using Distributions

@agent Sheep ContinuousAgent{2} begin
    speed::Float64
end


rng = MersenneTwister(123)
grass_level = 20
water_level = 1
mountain_level = 18
grass = BitArray(
    rand(rng, dims[1:2]...) .< ((grass_level .- heightmap) ./ (grass_level - water_level)),
)


dims[1:1]

function initialize_model(;
    n_sheep = 100, 
    extent = (100., 100.),
    speed = 1.0, 
    dt = 0.1, 
    grass_level = 20,
    water_level = 1, 
    mountain_level = 18, 
    seed = 123
)

    heightmap_url = "https://raw.githubusercontent.com/juliadynamics/" *
    "juliadynamics/master/videos/agents/runners_heightmap.jpg"
    ## download and load the heightmap. the grayscale value is converted to `float64` and
    ## scaled from 1 to 40
    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    ## the x and y dimensions of the pathfinder are that of the heightmap
    dims = (size(heightmap))
    ## the region of the map that is accessible to sheep is defined using `bitarrays`
    land_walkmap = BitArray(falses(dims...))
    water_map = BitArray(falses(dims...))

    # land walk map will be between `water_level` and `mountain_level`
    for i = 1:dims[1], j = 1:dims[2]
        if water_level < heightmap[i, j] < mountain_level
            land_walkmap[i, j] = true
        end
    end

    # water_map will be everything less or equal to `water_level`
    for i = 1:dims[1], j = 1:dims[2]
        if heightmap[i, j] ≤ water_level 
            water_map[i, j] = true
        end
    end



    rng = MersenneTwister(seed)

    space = ContinuousSpace(extent)

    ## Generate an array of random numbers, and threshold it by the probability of grass growing
    ## at that location. Although this causes grass to grow below `water_level`, it is
    ## effectively ignored by `land_walkmap`
    grass = BitArray(
        rand(rng, dims[1:2]...) .< ((grass_level .- heightmap) ./ (grass_level - water_level)),
    )



    properties = (
        speed = speed, 
        dt = dt, 
        grass = grass, 
        heightmap = heightmap
    )

    model = ABM(Sheep, space; rng, properties)


    for _ in 1:n_sheep
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(
            model, 
            vel, 
            # random_position(model), 
            speed
        )
    end


    # ### Add grass
    # for p in positions(model)
    #     if model.land_walkmap[p...] == 1
    #         fully_grown = rand(model.rng, Bool)
    #         countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1 
    #         model.countdown[p...] = countdown 
    #         model.fully_grown[p...] = fully_grown
    #     end
    # end 

    return model
end






model = initialize_model()        


function agent_step!(sheep, model)
    # move_agent!(sheep, model, sheep.speed)
    walk!(sheep, rand, model)
end






model = initialize_model()
fig, = abmplot(model)

fig



animalcolor(a) = :white

# We use `surface!` to plot the terrain as a mesh, and colour it using the `:terrain`
# colormap. Since the heightmap dimensions don't correspond to the dimensions of the space,
# we explicitly provide ranges to specify where the heightmap should be plotted.
function static_preplot!(ax, model)
    surface!(
        ax,
        (100/205):(100/205):100,
        (100/205):(100/205):100,
        model.heightmap;
        colormap = :terrain
    )
end


abmvideo(
    "rabbit_fox_hawk.mp4",
    model, agent_step!;
    figure = (resolution = (800, 700),),
    frames = 300,
    framerate = 15,
    ac = animalcolor,
    as = 1.0,
    static_preplot!,
    title = "Rabbit Fox Hawk with pathfinding"
)




model = initialize_model()
abmvideo(
    "test.mp4", model, agent_step!;
    framerate = 20, frames = 100
)