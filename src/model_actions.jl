"""
    make_model_stepping()

Creates a customized model step function `custom_model_step!` for an agent-based model (ABM). The function controls the behaviour of model properties. 


# Returns
- `custom_model_step! (function)`: A function that defines the model stepping behavior. This function takes an ABM as its only argument.

# Example

```julia
model_step! = make_model_stepping()
model_step!(model)
```
"""

function make_model_stepping(;
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




# function make_model_stepping(
#     pathfinding::Bool = true
# )
#     custom_model_step! = function(model::ABM)

#     if pathfinding
#         @inbounds for p in positions(model)
#             if model.land_walkmap[p...] == 1
#                 if !(model.fully_grown[p...])
#                     if model.countdown[p...] ≤ 0 
#                         model.fully_grown[p...] = true
#                         model.countdown[p...] = model.regrowth_time
#                     else
#                         model.countdown[p...] -= 1
#                     end
#                 end
#             end
#         end
#     else 
#         @inbounds for p in positions(model)
#             if !(model.fully_grown[p...])
#                 if model.countdown[p...] ≤ 0 
#                     model.fully_grown[p...] = true
#                     model.countdown[p...] = model.regrowth_time
#                 else
#                     model.countdown[p...] -= 1
#                 end
#             end
#         end
#     end
#     return custom_model_step!
# end

