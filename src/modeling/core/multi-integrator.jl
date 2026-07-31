struct NonlinearMultiDomainIntegrator2 <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    subintegrators::Dict{<: String, <: AbstractNonlinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        cellset = getcellset(grid, name)
        if first(sdh.cellset) ∈ cellset
            return setup_element_cache(subintegrator, sdh)
        end
    end
    return FerriteOperators.EmptyVolumetricElementCache()
end

# The subintegrators are keyed by *volumetric* subdomain name, so a subdomain is matched here exactly
# as in `setup_element_cache` above: the subintegrator that owns these cells also owns their weak
# boundary terms. Which facets of the subdomain actually carry a term is decided later, per facet, by
# `is_facet_in_cache`.
#
# This previously looked `name` up in the *surface* subdomains, which is a different namespace. It
# silently returned `EmptySurfaceElementCache()` — i.e. dropped every weak boundary condition — unless
# a facetset happened to share its name with the volumetric subdomain. The cuboid tests only pass
# through it because `generate_grid` names its facetsets "front"/"back", which collides with the
# cellset names those tests use.
function FerriteOperators.setup_boundary_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        cellset = getcellset(grid, name)
        if first(sdh.cellset) ∈ cellset
            return setup_boundary_cache(subintegrator, sdh)
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end

struct BilinearMultiIntegrator <: AbstractBilinearIntegrator
    subintegrators::Dict{<: String, <: AbstractBilinearIntegrator}
end

function FerriteOperators.setup_element_cache(
    integrator::BilinearMultiIntegrator,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        cellset = getcellset(grid, name)
        if first(sdh.cellset) ∈ cellset
            return setup_element_cache(subintegrator, sdh)
        end
    end
    return FerriteOperators.EmptyVolumetricElementCache()
end

function FerriteOperators.setup_boundary_cache(
    integrator::BilinearMultiIntegrator,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        has_surface_subdomain(grid, name) || continue
        surface_subdomain = grid.surface_subdomains[name]
        for facetset in values(surface_subdomain.data)
            cellset = first.(facetset)
            if first(sdh.cellset) ∈ cellset
                return setup_boundary_cache(subintegrator, sdh)
            end
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end


struct LinearMultiIntegrator <: AbstractLinearIntegrator
    subintegrators::Dict{<: String, <: AbstractLinearIntegrator}
end

function FerriteOperators.setup_element_cache(integrator::LinearMultiIntegrator, sdh::SubDofHandler)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        cellset = getcellset(grid, name)
        if first(sdh.cellset) ∈ cellset
            return setup_element_cache(subintegrator, sdh)
        end
    end
    return FerriteOperators.EmptyVolumetricElementCache()
end

function FerriteOperators.setup_boundary_cache(
    integrator::LinearMultiIntegrator,
    sdh::SubDofHandler,
)
    grid = get_grid(sdh.dh)
    for (name, subintegrator) in integrator.subintegrators
        has_surface_subdomain(grid, name) || continue
        surface_subdomain = grid.surface_subdomains[name]
        for facetset in values(surface_subdomain.data)
            cellset = first.(facetset)
            if first(sdh.cellset) ∈ cellset
                return setup_boundary_cache(subintegrator, sdh)
            end
        end
    end
    return FerriteOperators.EmptySurfaceElementCache()
end
