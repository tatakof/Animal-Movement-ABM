# Animal-Movement-ABM

## Agent-based Model for Animal Movement. 


### Code development Roadmap

This is the roadmap for developing the first ABMs, which to begin with will be single .jl files. 

- [x] v0.1. The space is `GridSpace`. Model has property `grass` (randomly distributed), with a simple time-based regrowth dynamic. Agents have `energy` and a `\Deltaenergy` fields. Agents move randomly (decreasing current energy levels) and eat grass when available (increasing current energy levels). 

- [x] v0.2. Add a home-range feature to the `agent_step!` function in order to avoid that sheep end up diffusing through space. This feature will consist in defining a point in space that will be the `attractor`, which will be the point where the sheep gravitate towards.  

- [x] v0.3. Add sheep gregarious behavior, in a similar way as implemented in the flocking [example](https://juliadynamics.github.io/Agents.jl/stable/examples/flock/) but adapted for a discrete space. Sheep will prefer regions or groups of patches that already contain sheep. Sheep will contain a field `visual_distance` which will define the distance to  which it can sense the surrounding sheep. 


- [x] v0.4. Implement directed movement with pathfinding. Now sheep will alternate between a random walk, a directed movement towards a random point in the `GridSpace` and a resting behaviour. There will be a `counter` parameter that dictates the amount of time-steps that each agent spends in each behaviour. When a behaviour ends, there's a transition to another behaviour. The probabilities to transition to another behaviour or to stay in the same behaviour are uniform. This implementation will follow the Mixed-Agent Ecosystem Pathfinding [example](https://juliadynamics.github.io/Agents.jl/stable/examples/rabbit_fox_hawk/)


- [x] v0.5. Add the spatial property `elevation`, which consists in a 2D matrix, where each value denotes the height of the terrain at that point.  Also define regions where sheep cannot walk and take them into account in the `agent_step!` function. 


- [x] v0.6. Use the `elevation` model property as a cost metric for the path finding.  


- [ ] v0.7. Agents will spend a variable amount of time in each behaviour that will be sampled from behaviour's time distributions. Once the time dedicated to a given behaviour ends, transition probability functions will determine which behaviour will be next. Each behaviour will be associated with different times of residence in the landscape cells and with different probabilities to choose a neighbouring cell. Both the times of residence and the probabilities to choose a neighbouring cell will be a function of the type of behaviour, of the agent's condition (e.g. energy) and of the landscape characteristics (height, slope, type of habitat, etc.). All of these implies setting a specific `counter` to each behaviour, that will be sampled from time distributions. It also implies defining 

nearby cells is not as straight forward to define in a continuous space
you can still enforce the concept of cells, by choosing

pathfinding can have the discretized version of the continuous space, so there you can have the nearby_walkable() function. 

To define a model property like grass in a continuous space, you have to add on top of it a discretized version. 

you have patches of grass but you move continously

get_spatial_property and get...property, these functions are made to work with discrete things in a continuous space. 

To make it truly generalizable you should make a function that makes functions using booleans. 


document the API?

--------------------------------------------------------------------------------------

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Animal-Movement-ABM

It is authored by Felici.

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "Animal-Movement-ABM"
```
which auto-activate the project and enable local path handling from DrWatson.
