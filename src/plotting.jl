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