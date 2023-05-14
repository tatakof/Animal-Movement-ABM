"""
    model_step!(model::ABM)

The `model_step!` function is responsible for updating the grass states in the model. It iterates through all positions in the model and performs the following steps:

1. If the grass at position `p` is not fully grown:
    - If the countdown at position `p` is less than or equal to 0, set the grass to fully grown and reset the countdown to the model's `regrowth_time`.
    - Otherwise, decrement the countdown at position `p` by 1.

# Arguments
- `model::ABM`: An instance of the agent-based model.

"""

function model_step!(model)
    @inbounds for p in positions(model)
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0 
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end