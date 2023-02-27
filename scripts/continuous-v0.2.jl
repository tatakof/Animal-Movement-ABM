# - [ ] v0.2. Add a home-range feature to the `agent_step!` function in order to 
# avoid that sheep end up diffusing through space. This feature will consist in 
# defining a point in space that will be the `attractor`, which will be the 
# point where the sheep gravitate towards.  



using Agents, LinearAlgebra
using Random
# import InteractiveDynamics
using InteractiveDynamics
using FileIO: load
using Distributions
using GLMakie

@agent Sheep ContinuousAgent{2} begin
    speed::Float64
end



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
        vel = (-1.0, -1.0) #Tuple(rand(model.rng, 2) * 2 .- 1)
        add_agent!(
            model, 
            vel, 
            # random_position(model), 
            speed
        )
    end


    ### Add grass
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



function agent_step!(sheep, model)
    vel1 = rand(truncated(Normal(), -1, 1))
    vel2 = rand(truncated(Normal(), -1, 1))
    sheep.vel = (vel1, vel2)
    move_agent!(sheep, model, sheep.speed)
    # walk!(sheep, rand, model)
    # walk!(sheep, (1.0, 1.0), model)
end


const sheep_polygon = Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    scale(rotate2D(sheep_polygon, φ), 2)
end


model = initialize_model()
fig, = abmplot(model; agent_step!, am = sheep_marker)

fig


rand(Uniform(-1, 1))


offset(a) = (-0.1, -0.1*rand()) 
ashape(a) = :circle 
acolor(a) = RGBAf(1.0, 1.0, 1.0, 0.8) 


grasscolor(model) = 30#model.countdown ./ model.regrowth_time

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
    # agent_step!, 
    # model_step!, 
    # params, 
    plotkwargs...
)
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













grasscolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (
    colormap = [:white, :green], 
    colorrange = (0, 1)
)

plotkwargs = (;
    as = 15,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = model.grass
)

params = Dict(
    :regrowth_time => 1:1:30
)

model = initialize_model()
fig, ax, abmobs = abmplot(model;
    # agent_step!, 
    # model_step!, 
    # params, 
    plotkwargs...
)
fig
