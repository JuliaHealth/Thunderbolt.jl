using Thunderbolt, CUDA
using Adapt

grid = generate_grid(Quadrilateral, (1,1))
gpugrid = Adapt.adapt_structure(CuVector, grid)

dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
close!(dh)
gpudh = Adapt.adapt_structure(CuVector, dh)
