using CairoMakie


xs = range(0, 10, length = 25)
ys = range(0, 15, length = 25)
zs = [cos(x) * sin(y) for x in xs, y in ys]

heatmap(xs, ys, zs, colormap = Reverse(:deep))


"
The best way forwards I think is to make a Pull Request at InteractiveDynamics.jl 
that allows this heatmap out of the box. Makie.jl supports a heatmap with arbitrary 
coordinates. So first one makes the coordinates, nbinx, nbiny = size(property). 
Then coordx = range(0, 1; length = nbinx) and same for y. 
Then, heatmap!(ax, coordx, coordy, property; ...) . Shouldn't be too hard. 
Have a go and if not possible for you I'll do it.
"


using CairoMakie
using Agents, InteractiveDynamics
using Random:bitrand

@agent Turtle ContinuousAgent{2} begin
	speed::Float64
end

function agent_step!( turtle, model)
	cs,sn = (x->(cos(x),sin(x)))((2rand()-1)*pi/15)
	turtle.vel = Tuple([cs sn;-sn cs]*collect(turtle.vel))
	move_agent!( turtle, model, turtle.speed)
end

function demo()
	dims = (30,30)
	world = ContinuousSpace(dims, spacing=1.0)
	model = ABM( Turtle, world; properties=Dict(:food => bitrand(dims)))
	foodmap(modl) = modl.food

	for _ in 1:50
		vel = (x->(cos(x),sin(x)))(2π*rand())
		add_agent!( model, vel, 1.0)
	end

	abmvideo(
		"Test.mp4", model, agent_step!;
		framerate = 50, frames = 200,
		ac=:blue, as=20, am=:circle,
		heatarray=foodmap,
	)
end

end

Base.size(cs::Agents.ContinuousSpace) = cs.dims

Test.demo()

