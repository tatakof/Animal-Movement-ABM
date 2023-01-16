
"
- [ ] v0.5. Add the spatial property `elevation`, which consists in a 
2D matrix, where each value denotes the height of the terrain at that point.
Also define regions where sheep cannot walk and take them into account in
the `agent_step!` function. 

"

## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions", "Plots", "ImageMagick", "FileIO"])

## Load packages
using Agents, Random
using GLMakie, InteractiveDynamics
import ImageMagick
using FileIO: load
using Agents.Pathfinding

## Agent definition
@agent Sheep GridAgent{2} begin
    energy::Float64
    # reproduction_prob::Float64
    Δenergy::Float64
    movement_cost::Float64
end

## Model function
function initialize_model(;
    n_sheep = 40, 
    regrowth_time = 30, 
    Δenergy_sheep = 4, 
    move_cost = 1, 
    water_level = 1, 
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
    space = GridSpace(dims, periodic = true)

    ### Model properties
    properties = (
        fully_grown = falses(dims), 
        countdown = zeros(Int, dims), 
        regrowth_time = regrowth_time, 
        elevation = elevation, 
        land_walkmap = land_walkmap, 
        landfinder = AStar(space; walkmap = land_walkmap)
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
                random_walkable(model, model.landfinder), 
                energy, 
                Δenergy_sheep, 
                move_cost
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

model = initialize_model()

## Agent stepping function

# Make a random walk taking into account the positions agents can
# and cannot take

function agent_step!(sheep, model)
    nearby_pos = [pos for pos in nearby_walkable(sheep.pos, model, AStar(model.space; walkmap = model.land_walkmap), 1)]
    move_agent!(sheep, nearby_pos[rand(1:length(nearby_pos))], model)
    sheep.energy -= sheep.movement_cost
    if sheep.energy < 0 
        kill_agent!(sheep, model)
        return
    end
    eat!(sheep, model)
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
        if model.land_walkmap[p...] == 1
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
