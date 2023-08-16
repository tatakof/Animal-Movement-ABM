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


include(srcdir("agent_actions.jl"))


@agent Sheep ContinuousAgent{2} begin #contains id, pos, and vel fields
    angle
    turn_angle
    mass::Float64
end




function initialize_model(;
    n_sheep = 1, 
    dims = (50, 50),
    seed = 123, 
    #VonMises_kappa = 0.5, 

)
    space2d = ContinuousSpace(dims;)
    rng = Random.MersenneTwister(seed)

    # properties = (
    #     :VonMises_kappa => 0.1
    # )

    mass = 1


    model = ABM(Sheep, space2d; rng, scheduler = Schedulers.Randomly())


    for _ in 1:n_sheep
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        angle = atan(vel[2], vel[1])
        turn_angle = 0 #initial turn angle is 0, since there's no movement yet
        add_agent!(
            model, 
            vel, 
            angle, 
            turn_angle, 
            mass
        )
    end
    

    return model

end

model = initialize_model(n_sheep = 1)



# https://juliadynamics.github.io/Agents.jl/stable/api/#Agents.randomwalk!
# Anything that supports rand can be used as an angle distribution instead. This can be useful to create correlated random walks.
function agent_step!(sheep, model)
    # randomwalk!(sheep, model; polar=VonMises(sheep.angle, model.properties.second))
    # custom_randomwalk!(sheep, model; polar=VonMises(sheep.angle, 9999))
    # custom_randomwalk!(sheep, model; polar=VonMises(atan(sheep.vel[2], sheep.vel[1]), 10))
    # correlated_randomwalk!(sheep, model; VonMises_kappa = 10000)
    # centrally_biased_randomwalk!(sheep, model; VonMises_kappa = 100)
    # biased_randomwalk!(sheep, model; bias = π)    
    # randomwalk!(sheep, model, rand(Weibull(2, 1)); polar=VonMises())
    # custom_randomwalk!(sheep, model, rand(Weibull(2, 1)); polar=VonMises(sheep.angle, 2))
    levy_walk(sheep, model)
end






factor = 0.6
const sheep_polygon = Polygon(Point2f[(-0.5 * factor, -0.5 * factor), (1 * factor, 0), (-0.5 * factor, 0.5 * factor)])

function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    InteractiveDynamics.scale(rotate2D(sheep_polygon, φ), 2)
end


fig, = abmplot(model; agent_step!)
fig


fig, = abmplot(model; agent_step!, am = sheep_marker)
fig



for id in 1:5
    agent = model[id]
    agent.mass = Inf
    agent.vel = Tuple(rand(2) * 2 .- 1) ./ 9999999999999999999
    # agent.vel = (0.00000, 0.00000)
end


model[5].vel


model[1].vel = Tuple(rand(2) * 2 .- 1) ./ 9999999999999999999




# Create a Weibull distribution
shape = 2
scale = 1
d = Weibull(shape, scale)

# Generate values for x-axis
x = range(0, stop=3, length=100)

# Calculate the corresponding probability density function (pdf) values
y = pdf.(d, x)

# Create the plot
lines(x, y)


rand(Weibull(2, 1))





factor = 0.6
const sheep_polygon = Polygon(Point2f[(-0.5 * factor, -0.5 * factor), (1 * factor, 0), (-0.5 * factor, 0.5 * factor)])

function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    InteractiveDynamics.scale(rotate2D(sheep_polygon, φ), 2)
end

adata = [(:angle, maximum), (:turn_angle, maximum)]
adata = [(:angle, mean), (:turn_angle, mean)]

fig, abmobs = abmexploration(model;
    agent_step!, 
    adata, 
    # am = sheep_marker
)

fig



fig, = abmplot(model; agent_step!)
fig


agent = model[1]
agent.vel
θ = rand(VonMises(0, 0.5))




using GLMakie

# const sheep_polygon = Makie.Polygon(Point2f[(-1, -1), (2, 0), (-1, 1)])

# function sheep_marker(b::Sheep)
#     φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
#     InteractiveDynamics.rotate_polygon(sheep_polygon, φ)
# end

# const sheep_polygon = Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])

factor = 0.6
const sheep_polygon = Polygon(Point2f[(-0.5 * factor, -0.5 * factor), (1 * factor, 0), (-0.5 * factor, 0.5 * factor)])

function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    InteractiveDynamics.scale(rotate2D(sheep_polygon, φ), 2)
end


# params = Dict(
#     :VonMises_kappa => 0.1:0.01:1.0
# )

fig, = abmplot(model; agent_step!, am = sheep_marker)


fig


adata = [(:angle, maximum), (:turn_angle, maximum)]
fig, abmobs = abmexploration(model;
    agent_step!, 
    adata, 
    # am = sheep_marker
)

fig








get_angle(vel) = atan(vel[2], vel[1])

# TEST turn angle computed
"df.turn_angle_test = compute_turn_angle(angle)"

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





x_id(a) = a.id==1
adata = [:pos, :vel, :angle, :turn_angle]
adata = [(:angle, maximum, x_id), (:turn_angle, maximum, x_id)]
adata = [(:angle, maximum), (:turn_angle, maximum)]


df = run!(model, agent_step!, 100; adata)

fig, abmobs = abmexploration(model;
    agent_step!, 
    adata, 
    # am = sheep_marker
)

fig


# Custom plots
fig, ax, abmobs = abmplot(model;
    agent_step!, 
    adata, figure = (; resolution = (1600,800)), am = sheep_marker
)
fig





abmobs


plot_layout = fig[:,end+1] = GridLayout()
count_layout = plot_layout[1,1] = GridLayout()

angle = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).maximum_angle))
turn_angle = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf).maximum_turn_angle))



ax_counts = Axis(count_layout[1,1];
    backgroundcolor = :lightgrey, ylabel = "Number of daisies by color")


scatterlines!(ax_counts, angle; color = :black, label = "black")
scatterlines!(ax_counts, turn_angle; color = :white, label = "white")

Legend(count_layout[1,2], ax_counts; bgcolor = :lightgrey)

ax_hist = Axis(plot_layout[2,1];
    ylabel = "Distribution of mean temperatures\nacross all time steps")
hist!(ax_hist, @lift($(abmobs.adf).maximum_turn_angle);
    bins = 50, color = :red,
    strokewidth = 2, strokecolor = (:black, 0.5),
)

fig

Agents.step!(abmobs, 1)
Agents.step!(abmobs, 1)
fig


autolimits!(ax_counts)
autolimits!(ax_hist)


on(abmobs.model) do m
    autolimits!(ax_counts)
    autolimits!(ax_hist)
end


for i in 1:100; step!(abmobs, 1); end
fig







# Other stuff

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




rand(Uniform(-π, π))


-π


# Plotting distributions

d = Levy()
p = Pareto()


# Generate values for x-axis
x = range(0, stop=3, length=100)

# Calculate the corresponding probability density function (pdf) values
y = pdf.(d, x)
y2 = pdf.(p, x)
fig, ax = lines(x, y)

lines!(ax, x, y2)


lines(x, pdf.(Pareto(1, 1),x))










using GLMakie
using StatsBase

# Generate some random data
θ = 2π * rand(1000)
r = rand(1000)

# Compute the histogram
bins = range(0, stop=2π, length=21)  # 20 bins
hist = fit(Histogram, θ, weights(r), bins)

# Convert bin edges to bin centers for plotting
bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) ./ 2

# Create a barplot in polar coordinates
fig = Figure()
ax = Axis(fig[1,1]; polar=true)
for (i, counts) in enumerate(hist.weights)
    b = bar!(ax, bin_centers[i], counts, width=2π/20)  # width of bars corresponds to the bin width
    b.color = (:blue, 0.5)
end
fig




#Polar Plots
#https://plotly.com/julia/polar-chart/
#https://goropikari.github.io/PlotsGallery.jl/src/rose.html

