# Domain management

Having multiple coupled subdomains is very common in multiphyics problems.
Furthermore it is also not uncommon to have mixed(-dimensional) grids, think e.g. about the Purkinje network and the myocardium in chamber electrophysiology simulations.
To manage these cases Thunderbolt.jl comes with some utilities. The first one is the [`SimpleMesh`](@ref), which is takes a [`Ferrite.Grid`] and extracts information about the subdomains. The subdomains are split up by element type to handle mixed grids properly.

This subdomain information can then be used to construct [`Ferrite.SubDofHandler`] to manage the field variables on subdomains:

```@docs
Thunderbolt.add_subdomain!
Thunderbolt.ApproximationDescriptor
```

Furthermore to manage data on subdomains we provide a non-uniform matrix-like data type.

```@docs
Thunderbolt.DenseDataRange
Thunderbolt.get_data_for_index
```

Two examples where this is used: The storate of element assembly and quadrature data on mixed grids.
