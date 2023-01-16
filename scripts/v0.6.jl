
# - [ ] v0.6. Use the `elevation` model property as a cost metric
#  for the path finding.  

## Install packages
# using Pkg
# Pkg.add(["Tables", "Random", "GLMakie", "InteractiveDynamics", "Distributions", "Plots", "ImageMagick", "FileIO"])

## Load packages
using Agents, Random
using GLMakie, InteractiveDynamics
import ImageMagick
using FileIO: load
using Agents.Pathfinding
using Distributions

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
    seed = 321, 
    counter = 50, 

)
    heightmap_url = "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/runners_heightmap.jpg"
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
        behav = zeros(Int, 1), 
        behav_counter = zeros(Int, 1),
        counter = counter,  
        elevation = elevation, 
        land_walkmap = land_walkmap, 
        landfinder = AStar(
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

    ### Set behaviour counter. 
    model.behav_counter[1] = counter


    return model
end

model = initialize_model()

## Agent stepping function
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
                random_walkable(model, model.landfinder), 
                model.landfinder
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
            move_along_route!(sheep, model, model.landfinder)
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


## Describe
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

# To view our starting population, we can build an overview plot using [`abmplot`](@ref).
# We define the plotting details for the wolves and sheep:
offset(a) = (-0.1, -0.1*rand()) 
ashape(a) = :circle 
acolor(a) = RGBAf(1.0, 1.0, 1.0, 0.8) 


# and instruct [`abmplot`](@ref) how to plot grass as a heatmap:
grasscolor(model) = model.countdown ./ model.regrowth_time
# homecolor(model) = model.home
# and finally define a colormap for the grass:
heatkwargs = (
    colormap = [:white, :green], 
    colorrange = (0, 1)
)

# and put everything together and give it to [`abmplot`](@ref)
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
    heatarray = model -> penaltymap(model.landfinder), 
    heatkwargs = (colormap  = :terrain,), 
    static_preplot!, 
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


