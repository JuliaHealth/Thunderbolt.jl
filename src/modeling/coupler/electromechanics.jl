function create_dh_helper(mesh, discretization, sym)
    ipc = _get_interpolation_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)
    return dh
end

struct ElectroMechanicalCoupler <: Thunderbolt.VolumeCoupler 
    ep_index::Int
    mech_index::Int
end

struct ElectroMechanicalCoupledModel{SM #=<: QuasiStaticModel =#, EM, CT <: ElectroMechanicalCoupler}
    ep_model::EM
    structural_model::SM
    coupler::CT
end

struct ElectroMechanicalSynchronizerCache{T<:Real, Dim, ∇uType<:DenseDataRange, I4Type<:DenseDataRange}
    Ca_mech::Vector{T}
    u_mech::Vector{Vec{Dim, T}}
    ∇u_mech::∇uType
    I4_mech::I4Type
    I4_vecs::Vector{Vector{T}}
end

struct ElectroMechanicalSynchronizer{NodalTransferOp1, NodalTransferOp2, T, Dim}
    transfer_op_EP_Mech::NodalTransferOp1
    transfer_op_Mech_EP::NodalTransferOp2
    cache::ElectroMechanicalSynchronizerCache{T, Dim}
end

function ElectroMechanicalSynchronizer(transfer_op_EP_Mech, transfer_op_Mech_EP)
    qrc=Thunderbolt.QuadratureRuleCollection(1)
    dh = transfer_op_EP_Mech.dh_to # mechanical dh
    dim = Ferrite.getrefdim(Ferrite.getfieldinterpolation(dh, (1,1)))
    Ca_mech = zeros(ndofs(dh))
    u_mech  = zeros(Vec{dim, Float64}, ndofs(dh))
    ∇u_mech = Thunderbolt.construct_qvector(Vector{Tensor{2, dim, Float64}}, Vector{Int64}, get_grid(dh), qrc)
    I4_mech = Thunderbolt.construct_qvector(Vector{Float64}, Vector{Int64}, get_grid(dh), qrc)
    I4_vecs = [zeros(getnquadpoints(getquadraturerule(qrc, cell))) for cell in getcells(get_grid(dh))] 
    ElectroMechanicalSynchronizer(
        transfer_op_EP_Mech,
        transfer_op_Mech_EP,
        ElectroMechanicalSynchronizerCache(
            Ca_mech ,
            u_mech  ,
            ∇u_mech ,
            I4_mech ,
            I4_vecs
        )
    )
end



function OS.forward_sync_external!(outer_integrator::OS.OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync::ElectroMechanicalSynchronizer)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    calcium_field_coeff = inner_integrator.cache.inner_solver_cache.op.integrator.volume_model.material_model.contraction_model.calcium_field
    calcium_state_idx = 4
    npoints = outer_integrator.subintegrator_tree[1][2].f.npoints
    calcium_in_EP = @view outer_integrator.subintegrator_tree[1][2].u[npoints * (calcium_state_idx-1)+1:npoints * calcium_state_idx]
    dh_mech = sync.transfer_op_EP_Mech.dh_to
    Ca_mech = sync.cache.Ca_mech
    Thunderbolt.transfer!(Ca_mech, sync.transfer_op_EP_Mech, calcium_in_EP)
    for sdh in dh_mech.subdofhandlers
        for cell in CellIterator(sdh)
            cell_idx = cellid(cell)
            for dof_idx in 1:length(celldofs(cell))
                calcium_field_coeff.elementwise_data[dof_idx, cell_idx] = Ca_mech[celldofs(cell)[dof_idx]]
            end
        end
    end
end

function OS.backward_sync_external!(outer_integrator::OS.OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync::ElectroMechanicalSynchronizer)    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    u_mech_flat = inner_integrator.u
    u_mech = sync.cache.u_mech
    dim = Ferrite.getrefdim(Ferrite.getfieldinterpolation(inner_integrator.f.dh, (1,1)))
    for i in 1:length(u_mech)
        u_mech[i] = Tensors.Vec(ntuple(j -> u_mech_flat[(i-1)*dim+j], Val(dim)))
    end
    dh = inner_integrator.f.dh
    qrc=Thunderbolt.QuadratureRuleCollection(1)
    ∇u_mech = sync.cache.∇u_mech
    ∇u_mech.data .= Ref(zero(eltype(∇u_mech.data)))
    integrator = Thunderbolt.BilinearDiffusionIntegrator(
        ConstantCoefficient(diagm(Tensor{2,dim}, 1.0)),
        qrc, # Allow e.g. mass lumping for explicit integrators.
        :displacement,
    )
    Thunderbolt.compute_quadrature_fluxes!(∇u_mech, sync.transfer_op_Mech_EP.dh_from, u_mech, :φₘ, integrator)
    fiber_coeff = inner_integrator.f.integrator.volume_model.material_model.microstructure_model.fiber_coefficient
    ∇u_mech.data .= Ref(diagm(Tensor{2,dim}, 1.0)) .+ ∇u_mech.data
    I4_mech = sync.cache.I4_mech
    proj = L2Projector(dh.grid)
    for sdh in dh.subdofhandlers
        fiber_coeff_cache = Thunderbolt.setup_coefficient_cache(fiber_coeff, Thunderbolt.getquadraturerule(qrc,sdh), sdh)
        qr = Thunderbolt.getquadraturerule(qrc,sdh)
        add!(proj, sdh.cellset, Thunderbolt.getinterpolation(LagrangeCollection{1}(), getcells(dh.grid, first(sdh.cellset))) ; qr_rhs = qr)
        for cell in CellIterator(sdh)
            cell_idx = cellid(cell)
            I4_mech_cell = Thunderbolt.get_data_for_index(I4_mech, cell_idx)
            F_cell = Thunderbolt.get_data_for_index(∇u_mech, cell_idx)
            for qp in QuadratureIterator(qr)
                f = evaluate_coefficient(fiber_coeff_cache, cell, qp, 0.0)
                I4_mech_cell[qp.i] = (F_cell[qp.i] ⋅ f) ⋅ (F_cell[qp.i] ⋅ f)
            end
        end
    end
    I4_vecs = sync.cache.I4_vecs
    cell_idx = 1
    for sdh in dh.subdofhandlers
        for cell in CellIterator(sdh)
            cell_idx = cellid(cell)
            I4_mech_cell = Thunderbolt.get_data_for_index(I4_mech, cell_idx)
            I4_vecs[cell_idx] .= I4_mech_cell
            cell_idx += 1
        end
    end
    close!(proj)
    projected = project(proj, I4_vecs)
    I4_in_EP = outer_integrator.subintegrator_tree[1][2].f.ode.I4
    Thunderbolt.transfer!(I4_in_EP, sync.transfer_op_Mech_EP, (projected))    
end

