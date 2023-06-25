"By the way, elevation could also be done in a 3D model as in the rabbit fox hawk example. However, I guess in our case it will be more performant to stick to 2D and use elevation as a cost metric for pathfinding." why would it be more performant to stick to 2D?


stuff: 

It also implies defining nearby cells is not as straight forward to define in a continuous space you can still enforce the concept of cells, by choosing

pathfinding can have the discretized version of the continuous space, so there you can have the nearby_walkable() function. 

To define a model property like grass in a continuous space, you have to add on top of it a discretized version. 

you have patches of grass but you move continously

get_spatial_property and get...property, these functions are made to work with discrete things in a continuous space. 

To make it truly generalizable you should make a function that makes functions using booleans. 


document the API?
