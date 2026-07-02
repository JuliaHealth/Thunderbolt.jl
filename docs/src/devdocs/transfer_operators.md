```@meta
DocTestSetup = :(using Thunderbolt)
```

# Transfer operators 

Thunderbolt implements the following field transfer methods:

## Nodal intergrid transfer

This works by evaluating the field value at the target mesh's nodes utilizing `Ferrite`'s `PointEvalHandler`.

!!! warning
    This method requires the target nodes to be overlap with the source mesh volume/surface. Nodes that fail this condition
    are assigned `NaN`.

This method is consturcted either using the provided convinience constructor [`NodalIntergridTransfer`](@ref) or by
using `NodalIntergridTransferStrategy()` as a transfer strategy for the construction of [`FieldTransferOperator`](@ref).

## Compactly-supported Radial Basis Function transfer

Fields can be transfered between non-matching potentially non-overlapping meshes by projecting them from
a finite element space to another space defined by global radial basis functions,
thus enabling extrapolation in the vacinity of the mesh boundary. This involves solving the linear
system $A \gamma = x$, where $x$ is the field in a finite element basis, $\gamma$ is the field
in radial basis function basis, and $A$ is the projection matrix constructed as

$$A_{ij} = φ(‖x_i, x_j‖ / r_j)$$

with $‖\circ, \dotsb‖$ being the distance measure used to calculate the distance between the two nodes,
$x_i$ being the coordinate of the $i$th node in the target mesh, $x_j$ being the
coordinate of the $j$th node in the source mesh, $r_j$ being the support radius for the $j$th
node in the source mesh, and $φ$ being the radial basis function of choice. 


### Distance measures

The internal transfer implementation supports two distance measures:

#### Euclidean distance

- `EuclideanDistanceMeasure(M, α)` is used for standard RBF transfer.
- `M` is the number of nearest neighbors used to determine the support radius.
- `α` is a scaling factor applied to the computed support radii.

#### Hybrid geodesic distance

- `GeodesicDistanceMeasure(M, α, β)` augments the Euclidean distance with mesh-geodesic information.
- The hybrid measure computes a short-path distance along the mesh graph and chooses between the
  Euclidean and geodesic distances based on a threshold.

The hybrid distance is defined internally as:

$$d(\circ, \square) = \begin{cases}
  d_{\mathrm{euc}}(\circ, \square), & d_{\mathrm{geo}}(\circ, \square) \leq d_{\mathrm{euc}}(\circ, \square) + \beta h_{\max}, \\
  d_{\mathrm{geo}}(\circ, \square), & d_{\mathrm{geo}}(\circ, \square) \gt d_{\mathrm{euc}}(\circ, \square) + \beta h_{\max}.\end{cases}$$

where $d_{\mathrm{euc}}(\circ, \square) = \lVert \circ - \square \rVert_{\text{L}^2}$ is the Euclidean norm, $d_{\mathrm{geo}}(\circ, \square)$ is the shortest-path mesh distance, $h_{\max}$ is the maximum edge length in the source mesh.

The parameter $\beta$ tunes whether Euclidean distance is preferred in regions where the geodesic distance is not significantly larger.

### Wendland radial basis functions

Thunderbolt includes functions for compactly supported Wendland kernels in 3D.
The implementation is represented by the internal type `WendlandRadialBasisFunction{3, k}`.
For `k = 0, 1, 2`, the kernel formulas are:

$$\phi(r) = \begin{cases}
  (1 - r)_+^2, & k = 0, \\
  (1 - r)_+^4 (1 + 4 r), & k = 1, \\
  (1 - r)_+^6 (35 r^2 + 18 r + 3), & k = 2.
\end{cases}$$

Here $(\circ)_+ = \text{max}(\circ, 0)$ is the positive part operator and $r$ is the normalized distance.

### Rescaling

The rescaled transfer path builds on the same compactly supported RBF basis used for plain
transfer, but adds a normalization step to preserve constant fields across meshes.

The rescaled method computes an additional normalization coefficient vector `γ_g` from
$$A \, \gamma_g = \mathbf{1},$$

where `\mathbf{1}` is a vector of ones. The transferred field is then evaluated as

$$\hat f(x) = \frac{\sum_j \gamma_{f,j} \phi\left(\frac{\lVert x - x_j \rVert}{r_j}\right)}{
  \sum_j \gamma_{g,j} \phi\left(\frac{\lVert x - x_j \rVert}{r_j}\right)}.$$

This rescaling ensures that a constant source field remains constant after transfer, which is
particularly important when the source and target meshes have different node densities.

## Source docstrings

```@docs
Thunderbolt.WendlandRadialBasisFunction
Thunderbolt.IntergridDofMapping
Thunderbolt.EuclideanDistanceMeasure
Thunderbolt.EuclideanDistanceMeasureCache
Thunderbolt.GeodesicDistanceMeasure
Thunderbolt.GeodesicDistanceMeasureCache
Thunderbolt.RadialBasisFunctionTransferStrategy
Thunderbolt.RescaledRadialBasisFunctionTransferStrategy
Thunderbolt.RadialBasisFunctionTransferOperator
Thunderbolt.RadialBasisFunctionGeodesicTransferOperator
Thunderbolt.RescaledRadialBasisFunctionTransferOperator
Thunderbolt.RescaledRadialBasisFunctionGeodesicTransferOperator
Thunderbolt.RadialBasisFunctionTransferStrategyCache
Thunderbolt.RescaledRadialBasisFunctionTransferStrategyCache
Thunderbolt.NodalIntergridTransferStrategy
Thunderbolt.NodalIntergridTransferStrategyCache
```
