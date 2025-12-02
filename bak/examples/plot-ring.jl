using Thunderbolt
using FerriteViz
import GLMakie
using JLD2

file = jldopen("./MidVentricularSectionHexG50-5-5_GHM-HO_AS1_RLRSQ75_Pelce.jld2")
solutionkeys = keys(file["timesteps"])
dh = file["dh"]
solutions = [file["timesteps/$(key)/solution"] for key in solutionkeys]
∇dh = first(FerriteViz.interpolate_gradient_field(dh, solutions[1], :u; copy_fields=[:u]))
∇u_over_time = [last(FerriteViz.interpolate_gradient_field(dh, u, :u; copy_fields=[:u])) for u ∈ solutions]

plotter = MakiePlotter(∇dh, ∇u_over_time[1])
# clip_plane = FerriteViz.ClipPlane(Vec((1.0,0.0,0.0)), 0.0)
# clipped_plotter = FerriteViz.crinkle_clip(plotter, clip_plane)
# fig, ax, sp = solutionplot(clipped_plotter,colormap=:plasma,deformation_field=:u,axis=(; show_axis=false))
# wireframe!(clipped_plotter,markersize=0,strokewidth=2,deformation_field=:u)

function F(u) # deformation gradient
    F = Tensor{2,3}(u[1:9])
    F += one(F)
    return F
end
Eᶠᶠproc(u) = tdot(F(u)) - one(F(u))

fig, ax, sp = solutionplot(plotter, colormap=:plasma, deformation_field=:u, axis=(; show_axis=false))
FerriteViz.wireframe!(plotter, markersize=0, strokewidth=2, deformation_field=:u)
GLMakie.record(fig, "ring.gif", ∇u_over_time, framerate=30) do solution
   FerriteViz.update!(plotter, solution)
end
