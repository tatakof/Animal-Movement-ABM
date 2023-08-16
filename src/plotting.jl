# include("distributions.jl")


"""
    plot_abm_model(model::ABM, agent_step!::Function, model_step!::Function)

Plot the Agent-Based Model (ABM) using the provided agent_step! and model_step! functions.

# Arguments
- `model::ABM`: The Agent-Based Model to be plotted.
- `agent_step!::Function`: A function describing the agent's step in the simulation.
- `model_step!::Function`: A function describing the model's step in the simulation.

# Returns
- `fig`: The figure object for the plot.
- `ax`: The axis object for the plot.
- `abmobs`: The ABMPlot object that contains the plot elements.

# Example
```julia
fig, ax, abmobs = plot_abm_model(model, agent_step!, model_step!)
```
"""



function plot_abm_model(model::ABM, agent_step!::Function, model_step!::Function)

    offset(a) = (-0.1, -0.1*rand(model.rng))
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

    fig, ax, abmobs = abmplot(model;
        agent_step!,
        model_step!,
        plotkwargs...
    )
    
    return fig, ax, abmobs
end


















function plot_weibull(k = 1.5, λ = 2.0)

    weibull_distribution = Weibull(k, λ)

    x = range(0, stop=8, length=1000)
    pdf_values = pdf.(weibull_distribution, x)

    fig = Figure(resolution = (800, 400))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "Density")
    lines!(ax, x, pdf_values)
    return fig
end




function plot_vonmises(μ = 0.0, k = 1.0)

    vonmises_distribution = VonMises(μ, κ)

    x = range(-π, stop=π, length=1000)
    pdf_values = pdf.(vonmises_distribution, x)

    fig = Figure(resolution = (800, 400))
    ax = Axis(fig[1, 1], xlabel = "Angle (radians)", ylabel = "Density")
    lines!(ax, x, pdf_values)


    return fig
end



# function plot_wrapcauchy(;μ = 0.0, ρ = 0.5)
#     #https://github.com/JuliaStats/Distributions.jl/pull/1665
#     #https://github.com/jeremyworsfold/Distributions.jl/blob/master/src/univariate/continuous/wrappedcauchy.jl
#     wrapcauchy_distribution = WrappedCauchy(μ, ρ)

#     x = range(-π, stop=π, length=1000)
#     pdf_values = pdf.(wrapcauchy_distribution, x)

#     fig = Figure(resolution = (800, 400))
#     ax = Axis(fig[1, 1], xlabel = "Angle (radians)", ylabel = "Density")
#     lines!(ax, x, pdf_values)
#     return fig
# end








function plot_shrubland(N, patch_size=2, density_factor=0.5)
    # Generate the shrubland
    shrubland = generate_shrubland(N, patch_size, density_factor)

    # Plot the shrubland as a heatmap
    fig = Figure()
    heatmap!(fig[1, 1], shrubland; colormap=:viridis)
    colorlegend!(fig[1, 1], "Density"; vertical=false, flipaxis=false)
    axis = fig[1, 1, Axis]
    axis.title = "Shrubland"
    return fig
end


function plot_shrubland2rand(N, num_spots = 5, σx = 5, σy = 5) 
    # Generate the shrubland
    shrubland = generate_shrubland2(N, num_spots, σx, σy) 
    custom_colormap = [:white, :green]

    # Plot the shrubland as a heatmap
    fig = heatmap(shrubland; colormap=custom_colormap)
    return fig

end

