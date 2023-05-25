"""
    grass_growth_dynamics(;walkmap::Bool = false)

Creates a custom model_step! function for an Agent-Based Model (ABM) simulating the growth dynamics of grass. 

The returned function operates on each position `p` in the model, managing grass growth based on walkability of land (if `walkmap` is true), and countdown/regrowth times.

# Arguments
- `walkmap::Bool=false`: Controls whether only walkable positions are considered for grass growth.

# Returns
- A custom model_step! function for the ABM.
"""

function grass_growth_dynamics(;
    walkmap::Bool = false 
)
    custom_model_step! = function(model::ABM)
        @inbounds for p in positions(model)
            if (!walkmap || model.land_walkmap[p...] == 1) && !(model.fully_grown[p...])
                if model.countdown[p...] ≤ 0 
                    model.fully_grown[p...] = true
                    model.countdown[p...] = model.regrowth_time
                else
                    model.countdown[p...] -= 1
                end
            end
        end
        return model
    end
    return custom_model_step!
end


