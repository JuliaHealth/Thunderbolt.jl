struct NonlinearMultiDomainIntegrator2 <: AbstractNonlinearIntegrator
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

function FerriteOperators.setup_boundary_cache(
    integrator::NonlinearMultiDomainIntegrator2,
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
