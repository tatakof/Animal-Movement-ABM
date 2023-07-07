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


model = initialize_model()


# https://juliadynamics.github.io/Agents.jl/stable/api/#Agents.randomwalk!
# Anything that supports rand can be used as an angle distribution instead. This can be useful to create correlated random walks.



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
adata = [:pos, :vel]




run!(model, agent_step!; adata)



can you get the angle from the randomwalk! function?

