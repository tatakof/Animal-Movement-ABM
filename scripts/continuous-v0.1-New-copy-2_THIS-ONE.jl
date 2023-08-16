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
include(srcdir("plotting.jl"))
include(srcdir("model_actions.jl"))

@agent Sheep ContinuousAgent{2} begin #contains id, pos, and vel fields
    angle
    turn_angle
    behaviour
    behav_counter
    speed
    energy
    status
end




function initialize_model(;
    n_sheep = 10, 
    dims = (100, 100),
    seed = 123, 
    #VonMises_kappa = 0.5, 
    # foraging_weibull_shape, 
    # foraging_weibull_scale, 
    # exploring_weibull_shape, 
    # exploring_weibull_scale, 
    # vonmises_concentration,
    # food_energy, 
    # moving_energy_loss, 
    # infection_prob, 


)
    space2d = ContinuousSpace(dims; periodic = false)
    rng = Random.MersenneTwister(seed)

    # properties = (
    #     :VonMises_kappa => 0.1
    # )

    mass = 1



    properties = (
        shrubland = generate_shrubland(dims[1])
    )


    model = ABM(Sheep, space2d; properties, rng, scheduler = Schedulers.Randomly())


    for _ in 1:n_sheep
        vel = Tuple(rand(model.rng, 2) * 2 .- 1)
        angle = atan(vel[2], vel[1])
        turn_angle = 0 #initial turn angle is 0, since there's no movement yet
        behav = rand([:Foraging, :Exploring, :Resting])
        speed = behav == :Foraging ? rand(abmrng(model), Weibull(5, 1)) : behav == :Exploring ? rand(abmrng(model), Weibull(5, 5)) : behav == :Resting ? 0.001 : 1000
        behav_counter = behav == :Foraging ? rand(abmrng(model), Weibull(5, 60)) : behav == :Exploring ? rand(abmrng(model), Weibull(5, 15)) : behav == :Resting ? 50 : 1000 # Esto vuela con recharge dynamics pero estaria bueno qe este, tal vez puede haber un mix donde algunos estadios tienen recharge dynamics y otros un tiempo de estar en ese estadio. # Gamma es la mas estandar para waiting times, weibull tambien pero no es tan estandar
        energy = 100
        status = :healthy # fix
        add_agent!(
            model, 
            vel, 
            angle, 
            turn_angle, 
            behav, 
            behav_counter, 
            speed, 
            energy, 
            status
        )
    end
    

    return model

end




model = initialize_model(n_sheep = 10)



# si forrajea en parche, no deberia salir del parche hasta que no salga del estado forrajeo. 




# bias correlated random walk para ir hasta donde esta la comida, es una posibilidad pero seria muy complicado encontrar 


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
    custom_randomwalk!(sheep, model, sheep.speed; polar=VonMises(sheep.angle, 0.0000001))
    sheep.behav_counter -= 1
    if sheep.energy >= 0 
        sheep.energy -= 0.05 
    end
    # levy_walk(sheep, model)



    if sheep.behav_counter == 0
        sheep.behaviour = rand([:Foraging, :Exploring, :Resting])
        sheep.behav_counter = sheep.behaviour == :Foraging ? rand(abmrng(model), Weibull(5, 60)) : sheep.behaviour == :Exploring ? rand(abmrng(model), Weibull(5, 15)) : sheep.behaviour == :Resting ? 50 : 1000
    end

    if sheep.behaviour == :Foraging 
        # eat!()
        if sheep.energy <= 200
            sheep.energy += 0.5
        end
        # if rand(Uniform()) <0.3
        #     sheep.status = rand([:infected]) #fix
        # end
    end

    if sheep.behaviour == :Foraging
        if rand(Uniform()) <0.5
            sheep.status = rand([:infected]) #fix
        end
    end
    # if agent.energy == 0
    #     kill!()
    # end
end


am(a) = a.behaviour == :Foraging ? :circle : a.behaviour == :Exploring ? :diamond : :rect
# as(a) = a.energy * 0.15

as(a) = a.energy * 1.00001
# as(a) = log(a.energy * 0.15)
# ac(a) = a.status == :healthy ? RGBAf(0.0, 1.0, 0.0, 0.8) : a.status == :infected ? RGBAf(0.0, 1.0, 0.0, 0.8) : RGBAf(1.0, 1.0, 1.0, 0.8)
ac(a) = a.status == :healthy ? :grey : a.status == :infected ? :red : RGBAf(1.0, 1.0, 1.0, 0.8)
fig, = abmplot(model; agent_step!, am = am, as = as, ac = ac)
fig



plot_shrubland3(1000, 1, 0.5)





# heatarray = model.properties
# heatkwargs = (
#     colormap = [:white, :green],
#     colorrange = (0, 1)
# )

# plotkwargs = (;
#     ac = ac, 
#     as = as, 
#     am = am, 
#     heatarray = model.properties, 
#     heatkwargs = heatkwargs
# )



# fig, = abmplot(model; agent_step!, plotkwargs...)
# fig





function plot_shrubland2(N, patch_size=2, density_factor=0.5)
    # Generate the shrubland
    shrubland = generate_shrubland(N, patch_size, density_factor)

    # Plot the shrubland as a heatmap
    fig = heatmap(shrubland; colormap=:viridis)
    return fig
end

N = 100
plot_shrubland2(N)


function plot_shrubland3(N, patch_size=2, density_factor=0.5)
    # Generate the shrubland
    shrubland = generate_shrubland(N, patch_size, density_factor)

    # Define a custom colormap where 0 maps to white and 1 maps to green
    # custom_colormap = [RGBf(1.0, 1.0, 1.0, 1.0), RGBf(0.0, 1.0, 0.0, 1.0)]
    custom_colormap = [:white, :green]

    # Plot the shrubland as a heatmap
    fig = heatmap(shrubland; colormap=custom_colormap)
    return fig
end


plot_shrubland3(1000, 1, 0.5)


model_step!() = nothing



agent_df, _ = run!(model, agent_step!, model_step!, n = 1000)


adata = [:angle, :turn_angle]#, :group]
adata = [:angle, :turn_angle, :behaviour, :behav_counter, :speed, :energy, :status]

model = initialize_model()
data, _ = run!(model, agent_step!, 5; adata)


vscodedisplay(data)




using Makie
using DataFrames

function plot_variables(data::DataFrame)
    # Getting the names of numerical variables
    variables_to_plot = [n for n in names(data) if eltype(data[!, n]) <: Number]

    # Array to store the figures
    figures = Figure[]

    for var in variables_to_plot
        # Getting the values for the variable
        values = data[!, var]
        
        # Creating a plot for each variable
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Step", ylabel = string(var), title = string(var))
        lines!(ax, data.step, values, color = :blue)

        # Add the figure to the array
        push!(figures, fig)
    end

    return figures
end











# Assuming `data` is your DataFrame, you can call this function to plot
figures = plot_variables(data)

using GLMakie
function plot_variables(data::DataFrame)
    # Array to store the figures
    figures = Figure[]

    # Iterate through all columns of the DataFrame
    for var in names(data)
        # Getting the values for the variable
        values = data[!, var]

        # Check if the variable is numerical
        if eltype(values) <: Number
            # Create a line plot for numerical variables
            fig = Figure()
            ax = Axis(fig[1, 1], xlabel = "Step", ylabel = string(var), title = string(var))
            lines!(ax, data.step, values, color = :blue)
        else
            # Create a histogram for categorical variables
            fig = Figure()
            ax = Axis(fig[1, 1], xlabel = string(var), title = string(var))
            hist(fig[1, 1], values, bins = 100)
        end

        # Add the figure to the array
        push!(figures, fig)
    end

    return figures
end

# Assuming `data` is your DataFrame, you can call this function to plot
figures = plot_variables(data)




custom_colormap = [RGBf(1.0, 1.0, 1.0, 1.0), RGBf(0.0, 1.0, 0.0, 1.0)]






factor = 0.6
const sheep_polygon = Polygon(Point2f[(-0.5 * factor, -0.5 * factor), (1 * factor, 0), (-0.5 * factor, 0.5 * factor)])

function sheep_marker(b::Sheep)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    InteractiveDynamics.scale(rotate2D(sheep_polygon, φ), 2)
end




fig, = abmplot(model; agent_step!)
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




using Distributions
using Makie

λ = 2.0  # Scale parameter
k = 1.5  # Shape parameter

weibull_distribution = Weibull(k, λ)

x = range(0, stop=8, length=1000)
pdf_values = pdf.(weibull_distribution, x)

fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "Density")
lines!(ax, x, pdf_values)
fig




include(srcdir("plotting.jl"))


μ = 0.0  # Mean or "location" parameter
κ = 1.0  # Concentration parameter

vonmises_distribution = VonMises(μ, κ)

x = range(-π, stop=π, length=1000)
pdf_values = pdf.(vonmises_distribution, x)

fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], xlabel = "Angle (radians)", ylabel = "Density")
lines!(ax, x, pdf_values)
fig


μ = 0.0  # Location parameter (mean direction)
ρ = 0.5  # Concentration parameter, must be in the range [0, 1)

wrapcauchy_distribution = WrappedCauchy(μ, ρ)

x = range(-π, stop=π, length=1000)
pdf_values = pdf.(wrapcauchy_distribution, x)

fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], xlabel = "Angle (radians)", ylabel = "Density")
lines!(ax, x, pdf_values)
fig




N = 50
num_spots = 5
σx = 5
σy = 5
shrubland = generate_shrubland2(N, num_spots, σx, σy)





plot_shrubland2(100)

