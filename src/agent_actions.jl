@enum WalkType RANDOM_WALK RANDOM_WALK_ATTRACTOR RANDOM_WALK_GREGARIOUS ALTERNATED_WALK RANDOM_WALKMAP


using Rotations
using StaticArrays: SVector, setindex






"""
    herbivore_dynamics(; walk_type::WalkType = RANDOM_WALK, eat::Bool = true, reproduce::Bool = true)

Returns a custom `agent_step!` based on the provided parameters.

# Arguments
- `walk_type::WalkType`: The type of walk the agent will perform. One of: 
    RANDOM_WALK, RANDOM_WALK_TO_ATTRACTOR OR RANDOM_WALK_GREGARIOUS
- `eat::Bool`: Whether the agent should eat or not.
- `reproduce::Bool`: Whether the agent should reproduce or not.
"""

function herbivore_dynamics(; 
    walk_type::WalkType = RANDOM_WALK, 
    eat::Bool = true, 
    reproduce::Bool = true
) 
    custom_agent_step! = function(
        agent::AbstractAgent, 
        model::ABM; 
        prob_random_walk = 0.9
    ) 

        # determine agent movement
        walk_ocurred = false
        if walk_type == RANDOM_WALK
            randomwalk!(agent, model)
            walk_ocurred = true
        elseif walk_type == RANDOM_WALK_ATTRACTOR
            execute_walk!(agent, model; walk_function = random_walk_to_attractor, prob_random_walk = prob_random_walk)
            walk_ocurred = true
        elseif walk_type == RANDOM_WALK_GREGARIOUS
            execute_walk!(agent, model; walk_function = random_walk_gregarious, prob_random_walk = prob_random_walk)
            walk_ocurred = true
        elseif walk_type == ALTERNATED_WALK
            execute_walk!(agent, model; walk_function = alternated_walk, prob_random_walk = 0)
            walk_ocurred = true
        elseif walk_type == RANDOM_WALKMAP
            execute_walk!(agent, model; walk_function = random_walkmap, prob_random_walk = 0)
            walk_ocurred = true
        end


        if walk_ocurred 
            reduce_energy!(agent, model, agent.movement_cost)
        end

        if eat
            eat!(agent, model; food = :grass)
        end

        if reproduce
            if rand(model.rng) ≤ agent.reproduction_prob
                reproduce!(agent, model)
            end
        end

    end
    return custom_agent_step!
end



"""
    reduce_energy!(agent, amount)

Reduces the energy of an agent by a specified amount. If the agent's energy goes below 0,
the agent is killed.

# Arguments
- `agent`: The agent whose energy is to be reduced.
- `amount`: The amount by which the agent's energy will be reduced.
"""
function reduce_energy!(agent, model, amount)
    agent.energy -= amount
    if agent.energy < 0
        Agents.kill_agent!(agent, model)
        return
    end
    return
end



"""
    execute_walk!(agent, model; walk_function, prob_random_walk)

Executes a specific walk function for an agent with a certain probability.
If the random draw is higher than the probability, performs a random walk.

# Arguments
- `agent`: The agent that will perform the walk.
- `model`: The model in which the agent exists.
- `walk_function`: The specific walk function to be executed.
- `prob_random_walk`: The probability of executing a random walk instead of the specific walk function.
"""

function execute_walk!(agent, model; walk_function, prob_random_walk)
    if rand(model.rng, Uniform(0, 1)) < prob_random_walk
        # "Normal" random walk
        randomwalk!(agent, model)
    else
        # Specific walk
        walk_function(agent, model)
    end
end



"""
    eat!(agent, model, food = :grass)

Updates the energy level of an agent by consuming food in a given model environment, and changes the food availability at the agent's position.


# Arguments
- `agent`: The agent performing the eating action.
- `model`: The agent-based model containing the agent and the environment information.
- `food`: An optional keyword argument representing the type of food (default :grass).

# Examples
```julia
agent = model[1] # Get the first agent in the model
eat!(agent, model; food = :grass) # Simulate the agent eating at its current position
```
"""

function eat!(agent::AbstractAgent, model::ABM; food::Symbol = :grass)
    if food == :grass
        if model.fully_grown[agent.pos...]
            agent.energy += agent.Δenergy
            model.fully_grown[agent.pos...] = false
        end
    end
    return
end



"""
    reproduce!(agent, model)

Simulates the reproduction of the given `agent` in the `model`. The agent's energy is divided in half, with the
newly created offspring receiving half of the parent's energy. The offspring is created as an instance of the
`Sheep` type with the same properties as the parent (except for the energy) and is placed at the same position
as the parent in the model.

This function modifies the agent's energy and adds a new agent to the model.

# Arguments
- `agent`: The agent performing the reproduction action.
- `model`: The agent-based model containing the agent and the environment information.

# Examples
```julia
agent = model[1] # Get the first agent in the model
reproduce!(agent, model) # Simulate the agent reproducing and creating an offspring
```

"""
function reproduce!(agent::AbstractAgent, model::ABM)
    agent.energy /= 2
    id = nextid(model)
    offspring = Sheep(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, agent.movement_cost, agent.visual_distance)

    add_agent_pos!(offspring, model)
end



function random_walkmap(agent::AbstractAgent, model::ABM)
    nearby_pos = [pos for pos in nearby_walkable(agent.pos, model, AStar(model.space; walkmap = model.land_walkmap), 1)]
    move_agent!(agent, nearby_pos[rand(model.rng, 1:length(nearby_pos))], model)

end





"""
    alternated_walk(agent::AbstractAgent, model::ABM)

The function represents an agent's movement behavior in the agent-based model (ABM) `model`. The agent alternates between three types of behaviors: random walk, directed movement towards a random point in the `GridSpace`, and resting. 

The behavior is controlled by a counter parameter in the model (`model.counter`). The agent spends an equal amount of time-steps in each behavior. When the time for a behavior ends (determined by the counter), the agent transitions to another behavior. The probabilities for transitioning to another behavior or staying in the current one are uniform.

The specific behaviors are as follows:

1. **Random Walk**: The agent performs a random walk in the `GridSpace`.
2. **Directed Walk**: The agent plans a route to a randomly selected walkable point in the `GridSpace` and follows the route.
3. **Rest**: The agent stays at its current position.

# Arguments

- `agent (AbstractAgent)`: The agent performing the alternated walk.
- `model (ABM)`: The agent-based model that contains the agent.

"""
function alternated_walk(agent::AbstractAgent, model::ABM)
    if model.behav_counter[1] == model.counter 
        model.behav[1] = sample(1:3)
        
        # If Directed movement, plan route
        if model.behav[1] == 2
            plan_route!(
                agent, 
                random_walkable(model, model.pathfinder), 
                model.pathfinder
            )
        end
    end

    if 0 < model.behav_counter[1] <= model.counter 
        # 1 == RandomWalk
        if model.behav[1] == 1
            randomwalk!(agent, model)
        # 2 == Directed Walk
        elseif model.behav[1] == 2
            move_along_route!(agent, model, model.pathfinder)
        # 3 == Rest
        elseif model.behav[1] == 3
            move_agent!(agent, agent.pos, model)
        end
        model.behav_counter[1] -= 1
    end

    if model.behav_counter[1] == 0
        model.behav_counter[1] = model.counter 
    end
end




"""
    random_walk_to_attractor(agent::AbstractAgent, model::ABM)

Perform a random walk for the agent towards an attractor point in the model. The attractor point is a location
towards which the agent gravitates. This function calculates the direction vector from the agent's position to the
attractor and moves the agent one step in that direction.

# Arguments

- `agent (AbstractAgent)`: An agent in the agent-based model.
- `model (ABM)`: The agent-based model containing the agent and the attractor point.

# Examples
TO COMPLETE

"""
function random_walk_to_attractor(agent::AbstractAgent, model::ABM)
    dist = collect(model.attractor .- agent.pos)
    direction = dist != [0, 0] ? round.(Int, normalize(dist)) : round.(Int, dist)
    walk!(agent, (direction...,), model, ifempty=true)
end







"""
    random_walk_gregarious(agent, model)

Performs a gregarious random walk for the given agent in the `model`. The function first determines
possible locations the current agent can move to and the neighboring agents within the current agent's visual distance. If there are
neighboring agents, the current agent moves towards the closest neighbor, considering the Euclidean distance between
them. The agent's movement is biased using a probability distribution based on the distance to the closest
neighbor, where closer locations have higher probabilities.

If there are no neighboring agents, the current agent performs a simple random walk, selecting a neighboring position
uniformly at random.

# Arguments
- `agent`: The agent to perform the random walk.
- `model`: The agent-based model containing the agents.

# Examples
```julia
agent = model[1] # Get the first agent in the model
random_walk_gregarious(agent, model) # Perform a gregarious random walk for the agent
```
"""
function random_walk_gregarious(agent::AbstractAgent, model::ABM)
    # Get the nearby locations that an agent can move to
    possible_locations = [pos for pos in nearby_positions(agent.pos, model)]

    # Get the ids of the nearby agents
    neighbor_ids = [ids for ids in nearby_ids(agent, model, agent.visual_distance)]

    # If there are neighboring agents, do a weighted walk toward the closest
    if length(neighbor_ids) > 0
        # Find the closest neighbor
        closest_neighbor_pos = nothing
        min_distance = Inf
        for ids in neighbor_ids
            neighbor_pos = model[ids].pos
            distance = euclidean_distance(neighbor_pos, agent.pos, model)
            if distance < min_distance
                min_distance = distance
                closest_neighbor_pos = neighbor_pos
            end
        end

        # Compute the euclidean distance of each neighboring position to the closest neighbor
        eudistance_to_closest_neighbor = [euclidean_distance(pos, closest_neighbor_pos, model) for pos in possible_locations]

        # Define function that computes the probs of moving in each direction
        f(x) = exp(-x / 3) # IMPROVEMENT: this function should be an argument

        # Compute the probs of moving in each direction according to the distance to the attractor
        probs_to_move = f.(eudistance_to_closest_neighbor) ./ sum(f.(eudistance_to_closest_neighbor))

        # now we sample the movements using the probs_to_move
        move_to = wsample(1:length(eudistance_to_closest_neighbor), probs_to_move)

        # and move towards that location
        move_agent!(agent, possible_locations[move_to], model)
    else
        # Do a random walk
        move_agent!(agent, possible_locations[rand(model.rng, 1:8)], model)
    end
end















########################################################################################################
## CONTINUOUS SPACE FUNCTIONS

########################################################################################################

#=
VonMises distribution

The Von Mises distribution, also known as the circular normal distribution or Tikhonov distribution, is a continuous probability distribution on the circle. It is often used in directional statistics. The Von Mises distribution is the circular analogue of the normal (or Gaussian) distribution.

In the context of directional statistics, the Von Mises distribution can be used to model angles, times of day, orientations, and any other measurements that can be represented as directions on a circle.


    Mu (µ), the mean direction, which corresponds to the "peak" of the distribution.
    Kappa (κ), the concentration parameter, which describes how strongly points cluster around the peak. The larger the value of κ, the more the distribution is concentrated around µ.

The probability density function (pdf) of the Von Mises distribution on the interval [-π, π] is given by:

f(x | µ, κ) = (1 / 2πI₀(κ)) * exp{κ cos(x - µ)}

where:

    I₀(κ) is the modified Bessel function of order 0.
    exp denotes the exponential function.
    cos is the cosine function.

This function describes the likelihood of a random variable distributed according to a Von Mises distribution taking on the value x. It has the property of being 2π-periodic, meaning that it repeats every 2π units.



In the basic Von Mises distribution:

    Mu (µ): This is the mean direction, and it's a scalar value representing an angle. It indicates the "center" of the distribution on the circle, corresponding to the most probable direction that samples will fall towards. The values are typically within the range of -π to π or 0 to 2π.

    Kappa (κ): This is the concentration parameter, and it's also a scalar value. It describes how concentrated or dispersed the distribution is around the mean direction. When kappa is close to zero, the distribution approaches a uniform distribution over the circle; when kappa is large, the distribution becomes tightly focused around the mean direction. Kappa is always a non-negative real number (κ ≥ 0).

Note: The Von Mises distribution can be extended into higher dimensions, which is known as the Von Mises-Fisher distribution. In that case, the mean direction µ becomes a vector on the unit hypersphere in the higher-dimensional space, but kappa remains a scalar.

=#


#=
WEIBULL

The Weibull distribution is a continuous probability distribution named after Wallodi Weibull, who described it in detail in 1951, although it was first identified by Fréchet (1927) and first applied by Rosin & Rammler (1933) to describe a particle size distribution.

This distribution is often used in survival analysis and reliability engineering, as it can model a variety of shapes of life-data, including increasing, decreasing, constant, and bathtub-shaped hazard functions.

Definition

The probability density function of the Weibull distribution is given by:

scss

f(x;λ,k) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k) for x ≥ 0, k > 0

where:

    λ > 0 is the scale parameter of the distribution
    k > 0 is the shape parameter.

If k = 1, the distribution reduces to the exponential distribution.

The cumulative distribution function is given by:

scss

F(x;λ,k) = 1 - exp(-(x/λ)^k) for x ≥ 0, k > 0

Interpretation

    The scale parameter λ is often referred to as the characteristic life. It determines the scale of the distribution function. Higher λ indicates a distribution with a longer tail (more variability).

    The shape parameter k is the Weibull slope or the Weibull modulus. It determines the shape of the distribution function. If k > 1, the failure rate increases with time (the older something is, the more likely it is to fail). If k < 1, the failure rate decreases with time (items are more likely to fail shortly after they are new, and less likely to fail as they age). If k = 1, the failure rate is constant over time.

Uses

The Weibull distribution is very versatile. It can mimic the behavior of other statistical distributions, including the normal and exponential distributions. This flexibility makes it widely used, particularly in the fields of reliability engineering and survival analysis, where it is used to model various types of life data, including human lifespans, life of components, failure times, and more.


=#
















"""
This function is a fork of the uniform random walk provided by Agents.jl. The main difference is that
    this fork tracks the angles of movement (using the inverse tangent) and the turning angles. 
    This function assumes that the agent has the fields `angle` and `turn_angle`
"""

"""
    uniform_randomwalk_mine!(agent::AbstractAgent, model::ABM{<:ContinuousSpace{D}}, r::Real=sqrt(sum(abs2.(agent.vel)))) where {D}

Modifies the velocity and angle of the given agent based on a random walk in a continuous space model. The new velocity is set to a random direction with magnitude `r`.

If the random vector's norm is zero, the function will generate a uniform random direction.

The agent's `angle` attribute will be updated to the angle of the new velocity, and the `turn_angle` attribute will be updated to the absolute difference between the new and previous angles.

Throws an `ArgumentError` if `r` is less than or equal to 0.

# Arguments
- `agent::AbstractAgent`: The agent to move.
- `model::ABM{<:ContinuousSpace{D}}`: The model in which the agent resides.
- `r::Real=sqrt(sum(abs2.(agent.vel)))`: The magnitude of the agent's displacement. Defaults to the magnitude of the agent's current velocity.

# Returns
- Nothing. This function modifies the input agent in-place.
"""



# https://github.com/JuliaDynamics/Agents.jl/blob/2894fbbf0de3f090cf9e9dcec80f201ac0f0d644/src/spaces/walk.jl#L340
function custom_uniform_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{D}},
    r::Real=sqrt(sum(abs2.(agent.vel)))
) where {D}
    if r ≤ 0
        throw(ArgumentError("The displacement must be larger than 0."))
    end
    rng = abmrng(model)
    dim = Val(D)
    v = ntuple(_ -> randn(rng), dim)
    norm_v = sqrt(sum(abs2.(v)))
    if !iszero(norm_v)
        direction = v ./ norm_v .* r
    else
        direction = ntuple(_ -> rand(rng, (-1, 1)) * r / sqrt(D), dim)
    end
    # angle based on previous agent.vel
    previous_angle = atan(agent.vel[2], agent.vel[1])
    # assigns new velocity
    agent.vel = direction
    # angle based on new agent.vel
    current_angle = atan(agent.vel[2], agent.vel[1])
    agent.angle = current_angle
    agent.turn_angle = abs(diff([previous_angle, current_angle])[1])
    walk!(agent, direction, model)
end


# TO DO Multiple dispatch with speed distribution?
function custom_uniform_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{D}},
    r::Real=sqrt(sum(abs2.(agent.vel)))
) where {D}
    if r ≤ 0
        throw(ArgumentError("The displacement must be larger than 0."))
    end
    rng = abmrng(model)
    dim = Val(D)
    v = ntuple(_ -> randn(rng), dim)
    norm_v = sqrt(sum(abs2.(v)))
    if !iszero(norm_v)
        direction = v ./ norm_v .* r
    else
        direction = ntuple(_ -> rand(rng, (-1, 1)) * r / sqrt(D), dim)
    end
    # angle based on previous agent.vel
    previous_angle = atan(agent.vel[2], agent.vel[1])
    # assigns new velocity
    agent.vel = direction
    # angle based on new agent.vel
    current_angle = atan(agent.vel[2], agent.vel[1])
    agent.angle = current_angle
    agent.turn_angle = abs(diff([previous_angle, current_angle])[1])
    walk!(agent, direction, model)
end



# function custom_randomwalk!(
#     agent::AbstractAgent,
#     model::ABM{<:ContinuousSpace{2}};
#     polar=nothing,
# )
#     if isnothing(polar)
#         throw(ArgumentError("Must provide a value for the argument `polar`."))
#     end

#     θ = rand(abmrng(model), polar)
#     agent.turn_angle = θ

#     direction = Tuple(rotate(SVector(agent.vel), θ))
#     agent.vel = direction
    
#     agent.angle = atan(agent.vel[2], agent.vel[1])


#     walk!(agent, direction, model)
# end


"""
    randomwalk!(agent, model::ABM{<:ContinuousSpace} [, r];
        [polar=Uniform(-π,π), azimuthal=Arccos(-1,1)]
    )

Re-orient and move `agent` for a distance `r` in a random direction
respecting space boundary conditions. By default `r = norm(agent.vel)`.

The `ContinuousSpace` version is slightly different than the grid space.
Here, the agent's velocity is updated by the random vector generated for
the random walk. 

Uniform/isotropic random walks are supported in any number of dimensions
while an angles distribution can be specified for 2D and 3D random walks.
In this case, the velocity vector is rotated using random angles given by 
the distributions for polar (2D and 3D) and azimuthal (3D only) angles, and 
scaled to have measure `r`. After the re-orientation the agent is moved for 
`r` in the new direction.

Anything that supports `rand` can be used as an angle distribution instead. 
This can be useful to create correlated random walks.
"""
# Only with constant speed
function custom_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}},
    r::Real;
    polar=nothing,
)
    if isnothing(polar)
        return custom_uniform_randomwalk!(agent, model, r)
    end
    if r ≤ 0
        throw(ArgumentError("The displacement must be larger than 0."))
    end



    θ = rand(abmrng(model), polar)

    agent.turn_angle = θ

    relative_r = r/LinearAlgebra.norm(agent.vel) #the norm of agent.vel is equal to agent.speed
    direction = Tuple(rotate(SVector(agent.vel), θ)) .* relative_r
    # agent.vel = direction 
    agent.vel = direction .* agent.speed

    agent.angle = atan(agent.vel[2], agent.vel[1])

    walk!(agent, direction, model)
end




# speed comes from a prob distribution
function custom_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}},
    r::Real;
    polar=nothing,
    speed_distribution=nothing
)
    if isnothing(polar)
        return custom_uniform_randomwalk!(agent, model, r)
    end
    if r ≤ 0
        throw(ArgumentError("The displacement must be larger than 0."))
    end


    # if !isnothing(speed_distribution)
    #     agent.speed = rand(abmrng(model), speed_distribution)
    # end

    # 
    agent.speed = agent.behaviour == :Foraging ? rand(abmrng(model), Weibull(5, 1)) : agent.behaviour == :Exploring ? rand(abmrng(model), Weibull(5, 5)) : agent.behaviour == :Resting ? 0.001 : 1000


    θ = rand(abmrng(model), polar)

    agent.turn_angle = θ

    relative_r = r/LinearAlgebra.norm(agent.vel) #the norm of agent.vel is equal to agent.speed
    direction = Tuple(rotate(SVector(agent.vel), θ)) .* relative_r
    # agent.vel = direction 
    agent.vel = direction .* agent.speed

    agent.angle = atan(agent.vel[2], agent.vel[1])

    walk!(agent, direction, model)
end



















"""

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
All the specific randomwalk functions (such as correlated, biased, etc) will be functions that call 
custom_randomwalk! with different parameters

"""



"""
Correlated Random Walk (CRW)

El Correlated Random Walk intenta modelar la persistencia en la dirección de movimiento. 

Von Mises tiene un parámetro de media Mu. Para hacer un CRW, mu debe tomar el valor del angulo de direccion previo. O alternativamente, al angulo de direccion previo tenes que sumarle un valor de una Von Mises centrada en 0.  
Von mises también tiene un parámetro Kappa, que determina el grado de concentración alrededor de la media. 

"""


function correlated_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}};
    VonMises_kappa = 1
)
    θ = rand(abmrng(model), VonMises(agent.angle, VonMises_kappa))
    agent.turn_angle = θ

    direction = Tuple(rotate(SVector(agent.vel), θ))
    agent.vel = direction
    
    agent.angle = atan(agent.vel[2], agent.vel[1])


    walk!(agent, direction, model)

end





"""
Biased Random Walk (BRW)
El animal siempre se mueve en una dirección. Esta dirección es la media de la distribución circular. 

"""


function biased_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}};
    VonMises_kappa = 1, 
    bias = 1 # -π to π?
)
    θ = rand(abmrng(model), VonMises(bias, VonMises_kappa))
    agent.turn_angle = θ

    direction = Tuple(rotate(SVector(agent.vel), θ))
    agent.vel = direction
    
    agent.angle = atan(agent.vel[2], agent.vel[1])


    walk!(agent, direction, model)

end






"""
## Centrally Biased Random Walk (CBRW)
El animal sabe hacia qué punto quiere ir. Ese punto está en una dirección respecto a la posición actual del animal. Por lo tanto, en cada t, hay que calcular la dirección al punto al que el animal quiere ir, y usar esa dirección como media de la distribución circular.   

"""



function centrally_biased_randomwalk!(
    agent::AbstractAgent,
    model::ABM{<:ContinuousSpace{2}};
    VonMises_kappa = 1, 
    center_of_attraction = [25, 25]
)
    θ = rand(abmrng(model), VonMises(atan(center_of_attraction[1], center_of_attraction[2]), VonMises_kappa))
    agent.turn_angle = θ

    direction = Tuple(rotate(SVector(agent.vel), θ))
    agent.vel = direction
    
    agent.angle = atan(agent.vel[2], agent.vel[1])


    walk!(agent, direction, model)

end





# """
# ## Biased Correlated Random Walk (BCRW)
# Ahora en cada t hay dos direcciones, la dirección previa (por la parte de Correlated) y la dirección a la que quiere ir (por el Biased). Ahora hay un parámetro que indica el peso entre la importancia del Bias y la importancia del Correlated. Hay que hacer como un promedio ponderado entre la dirección del Bias y la dirección del Correlated, pero no es tan sencillo porque son vectores?





# """



# function biased_correlated_random_walk!()


# end








"""
A Levy flight, or Levy walk, is a type of random walk where the step-lengths have a probability distribution that is heavy-tailed. When this step-length distribution is a power-law distribution, the random walk is known as a Levy flight.


The Lévy distribution, also known as the Lévy alpha-stable distribution, is a type of probability distribution that has heavy tails. This distribution is named after the French mathematician Paul Lévy.

The Lévy distribution is "stable", meaning that the sum of two independent random variables with this distribution also has a Lévy distribution. It is part of a broader class of distributions known as stable distributions, which also includes the normal (or Gaussian) distribution. However, unlike the normal distribution, the Lévy distribution doesn't have a finite variance.

In other words, the Lévy distribution is capable of producing occasional values that are surprisingly far from the mean, a property that has led to its use in modeling phenomena like financial market returns and the movement patterns of animals, where such "Lévy flights" can occur.

The probability density function of a Lévy distribution (with location parameter μ and scale parameter c) is given by:

```scss

f(x;μ,c) = sqrt(c / (2π)) * exp( -c / (2*(x-μ))) / (x - μ)^(3/2),  for x > μ
```

Here, μ is the location parameter (similar to the mean in a normal distribution), and c is the scale parameter (similar to the standard deviation in a normal distribution).

It's also worth noting that the Lévy distribution is a type of "power-law" distribution. This means that the probability of drawing a value x from the distribution decreases as a power of x. In other words, large values are unlikely, but less so than in distributions like the normal distribution that decay exponentially.




Figure 1. Examples illustrating simulated URWs (a), BRWs (b) and LWs (c) of a single individual in two spatial dimensions. For the URWs, the individual chooses its movement direction and angle from a uniform distribution and moves a constant step at each time ( pr ¼ pl and 12pr2pl is the probability of waiting). For the BRWs, we set the probability distribution of the movement directions such that the individual is more likely to move left ( pl . pr and 1 2 pr – pl is the probability of waiting). For the LWs, the individual chooses its movement direction and the angle from a uniform distribution but the step length is chosen from a heavy-tailed distribution (Pareto distribution with infinite variance). For each simulated type of movement, the mean step length is equal.
"""



function levy_walk(agent, model; pareto_shape = 1, pareto_scale = 1)
    custom_randomwalk!(agent, model, rand(Pareto(pareto_shape, pareto_scale)); polar=Uniform(-π, π))
end






"""
    rotate(w::SVector{2}, θ::Real)
Rotate two-dimensional vector `w` by an angle `θ`.
The angle must be given in radians.
"""
rotate(w::SVector{2}, θ::Real) = Angle2d(θ) * w


















###########################################################################################
#PLOTTING
using DataStructures: CircularBuffer





"""
    abmexploration(model::ABM; alabels, mlabels, kwargs...)

Open an interactive application for exploring an agent based model and
the impact of changing parameters on the time evolution. Requires `Agents`.

The application evolves an ABM interactively and plots its evolution, while allowing
changing any of the model parameters interactively and also showing the evolution of
collected data over time (if any are asked for, see below).
The agent based model is plotted and animated exactly as in [`abmplot`](@ref),
and the `model` argument as well as splatted `kwargs` are propagated there as-is.
This convencience function *only works for aggregated agent data*.

Calling `abmexploration` returns: `fig::Figure, abmobs::ABMObservable`. So you can save 
and/or further modify the figure and it is also possible to access the collected data 
(if any) via the `ABMObservable`.

Clicking the "reset" button will add a red vertical line to the data plots for visual
guidance.

## Keywords arguments (in addition to those in `abmplot`)
* `alabels, mlabels`: If data are collected from agents or the model with `adata, mdata`,
  the corresponding plots' y-labels are automatically named after the collected data.
  It is also possible to provide `alabels, mlabels` (vectors of strings with exactly same
  length as `adata, mdata`), and these labels will be used instead.
* `figure = NamedTuple()`: Keywords to customize the created Figure.
* `axis = NamedTuple()`: Keywords to customize the created Axis.
* `plotkwargs = NamedTuple()`: Keywords to customize the styling of the resulting
  [`scatterlines`](https://makie.juliaplots.org/dev/examples/plotting_functions/scatterlines/index.html) plots.
"""



function custom_abmexploration(model;
        figure = NamedTuple(),
        axis = NamedTuple(),
        alabels = nothing,
        mlabels = nothing,
        plotkwargs = NamedTuple(),
        kwargs...
    )
    fig, ax, abmobs = custom_abmplot(model; figure, axis, kwargs...)
    abmplot_object = ax.scene.plots[1]

    adata, mdata = abmobs.adata, abmobs.mdata
    !isnothing(adata) && @assert eltype(adata)<:Tuple "Only aggregated agent data are allowed."
    !isnothing(alabels) && @assert length(alabels) == length(adata)
    !isnothing(mlabels) && @assert length(mlabels) == length(mdata)
    custom_init_abm_data_plots!(fig, abmobs, adata, mdata, alabels, mlabels, plotkwargs, abmplot_object.stepclick, abmplot_object.resetclick)
    return fig, abmobs
end


function custom_abmplot(model::Agents.ABM;
        figure = NamedTuple(),
        axis = NamedTuple(),
        kwargs...)
    fig = Figure(; figure...)
    ax = fig[1,1][1,1] = agents_space_dimensionality(model) == 3 ?
        Axis3(fig; axis...) : Axis(fig; axis...)
    abmobs = custom_abmplot!(ax, model; kwargs...)

    return fig, ax, abmobs
end



function custom_abmplot!(ax, model::Agents.ABM;
        # These keywords are given to `ABMObservable`
        agent_step! = Agents.dummystep,
        model_step! = Agents.dummystep,
        adata = nothing,
        mdata = nothing,
        when = true,
        kwargs...)
    abmobs = ABMObservable(model; agent_step!, model_step!, adata, mdata, when)
    abmplot!(ax, abmobs; kwargs...)

    return abmobs
end





function custom_init_abm_data_plots!(fig, abmobs, adata, mdata, alabels, mlabels, plotkwargs, stepclick, resetclick)
    La = isnothing(adata) ? 0 : size(abmobs.adf[])[2]-1
    Lm = isnothing(mdata) ? 0 : size(abmobs.mdf[])[2]-1
    La + Lm == 0 && return nothing # failsafe; don't add plots if dataframes are empty

    plotlayout = fig[:, end+1] = GridLayout(tellheight = false)
    axs = []

    for i in 1:La # add adata plots
        y_label = string(adata[i][2]) * "_" * string(adata[i][1])
        points = @lift(Point2f.($(abmobs.adf).step, $(abmobs.adf)[:,y_label]))
        ax = plotlayout[i, :] = Axis(fig)
        push!(axs, ax)
        ax.ylabel = isnothing(alabels) ? y_label : alabels[i]
        c = JULIADYNAMICS_COLORS[mod1(i, 3)]
        scatterlines!(ax, points;
            color = c, strokecolor = c, strokewidth = 0.5,
            label = ax.ylabel, plotkwargs...
        )
    end

    for i in 1:Lm # add mdata plots
        y_label = string(mdata[i])
        points = @lift(Point2f.($(abmobs.mdf).step, $(abmobs.mdf)[:,y_label]))
        ax = plotlayout[i+La, :] = Axis(fig)
        push!(axs, ax)
        ax.ylabel = isnothing(mlabels) ? y_label : mlabels[i]
        c = JULIADYNAMICS_COLORS[mod1(i+La, 3)]
        scatterlines!(ax, points;
            color = c, strokecolor = c, strokewidth = 0.5,
            label = ax.ylabel, plotkwargs...
        )
    end

    if La+Lm > 1
        for ax in @view(axs[1:end-1]); hidexdecorations!(ax, grid = false); end
    end
    axs[end].xlabel = "step"

    # Trigger correct, and efficient, linking of x-axis
    linkxaxes!(axs[end], axs[1:end-1]...)
    on(stepclick) do clicks
        xlims!(axs[1], Makie.MakieLayout.xautolimits(axs[1]))
        for ax in axs
            ylims!(ax, Makie.MakieLayout.yautolimits(ax))
        end
    end
    on(resetclick) do clicks
        for ax in axs
            vlines!(ax, [abmobs.s.val], color = "#c41818")
        end
    end
    return nothing
end


    # x2, y2 = 1, 2

    # tail = 300

    # traj = CircularBuffer{Point2f}(tail)

    # fill!(traj, Point2f(x2, y2))

    # traj = Observable(traj)

    # c = to_color(:purple)

    # tailcol = [RGBAf(c.r, c.g, c.b, (i/tail)^2) for i in 1:tail]
    # lines!(ax, traj; linewidth = 3, color = tailcol)    























###############


