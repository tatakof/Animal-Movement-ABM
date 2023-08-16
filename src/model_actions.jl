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













"""
    generate_shrubland(N, patch_size=2, density_factor=0.5)

Create a N x N grid representing shrubland, with 1s and 0s. 

# Arguments
- `N::Int`: The size of the grid (N x N).
- `patch_size::Int`: The size of the patches that will be either filled with shrubs or empty (default is 2).
- `density_factor::Float64`: The proportion of patches that will contain shrubs (default is 0.5).

# Returns
- `Array{Int}`: A grid of N x N with 1s representing shrubs and 0s representing empty cells.

# Example
```julia
N = 10
shrubland = generate_shrubland(N)
println(shrubland)
"""



function generate_shrubland(N, patch_size=2, density_factor=0.5)
    shrubland = Array{Int}(undef, N, N)
    for i in 1:patch_size:N
        for j in 1:patch_size:N
            patch = rand() < density_factor ? 1 : 0
            for di in 0:patch_size-1
                for dj in 0:patch_size-1
                    if i + di <= N && j + dj <= N
                        shrubland[i + di, j + dj] = patch
                    end
                end
            end
        end
    end
    return shrubland
end















"""
    generate_shrubland(N, num_spots, σx, σy, threshold=0.5)

Generate a shrubland grid of size `N x N` with `num_spots` spots of shrubs.
The spots are distributed following a bivariate Gaussian function.

# Arguments
- `N::Int`: The size of the grid (N x N).
- `num_spots::Int`: The number of Gaussian spots to create.
- `σx::Float64`: The standard deviation in the x direction for the Gaussian spots.
- `σy::Float64`: The standard deviation in the y direction for the Gaussian spots.
- `threshold::Float64`: The threshold above which cells will contain shrubs (default is 0.5).

# Returns
- `Array{Int, 2}`: The generated shrubland grid.
"""
function generate_shrubland2(N, num_spots, σx, σy, threshold=0.5)
    shrubland = zeros(N, N)
    for _ in 1:num_spots
        # Randomly select the center of a Gaussian spot
        x0 = rand(1:N)
        y0 = rand(1:N)

        # Apply the Gaussian function to cells around the center
        for i in 1:N
            for j in 1:N
                p = gaussian(i, j, x0, y0, σx, σy)
                if p > threshold
                    shrubland[i, j] = 1
                end
            end
        end
    end
    return shrubland
end

function gaussian(x, y, x0, y0, σx, σy)
    return exp(-((x - x0)^2 / (2 * σx^2) + (y - y0)^2 / (2 * σy^2)))
end
