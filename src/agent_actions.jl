@enum WalkType RANDOM_WALK RANDOM_WALK_ATTRACTOR RANDOM_WALK_GREGARIOUS ALTERNATED_WALK




"""
    make_agent_stepping(; walk_type::WalkType = RANDOM_WALK, eat::Bool = true, reproduce::Bool = true)

Returns a custom `agent_step!` based on the provided parameters.

# Arguments
- `walk_type::WalkType`: The type of walk the agent will perform. One of: 
    RANDOM_WALK, RANDOM_WALK_TO_ATTRACTOR OR RANDOM_WALK_GREGARIOUS
- `eat::Bool`: Whether the agent should eat or not.
- `reproduce::Bool`: Whether the agent should reproduce or not.
"""

function make_agent_stepping(; 
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
