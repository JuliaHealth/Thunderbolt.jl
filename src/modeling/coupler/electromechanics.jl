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

struct ElectroMechanicalSynchronizer{NodalTransferOp1, NodalTransferOp2}
    transfer_op_EP_Mech::NodalTransferOp1
    transfer_op_Mech_EP::NodalTransferOp2
end

function OS.forward_sync_external!(outer_integrator::OS.OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync::ElectroMechanicalSynchronizer)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    calcium_field_coeff = inner_integrator.cache.inner_solver_cache.op.integrator.volume_model.material_model.contraction_model.calcium_field
    calcium_state_idx = 4
    npoints = outer_integrator.subintegrator_tree[1][2].f.npoints
    calcium_in_EP = outer_integrator.subintegrator_tree[1][2].u[npoints * (calcium_state_idx-1)+1:npoints * calcium_state_idx]
    dh_mech = sync.transfer_op_EP_Mech.dh_to
    Ca_mech = zeros(ndofs(dh_mech))
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
    u_mech = [Tensors.Vec((u_mech_flat[i:i+2]...)) for i in 1:3:length(u_mech_flat)]
    dh = inner_integrator.f.dh
    qrc=Thunderbolt.QuadratureRuleCollection(1)
    dim = Ferrite.getrefdim(Ferrite.getfieldinterpolation(inner_integrator.f.dh, (1,1)))
    ∇u_mech = Thunderbolt.construct_qvector(Vector{Tensor{2, dim}}, Vector{Int64}, dh.grid, qrc)
    I=SymmetricTensor{2,3}((1.0,0.0,0.0,1.0,0.0,1.0))
    integrator = Thunderbolt.BilinearDiffusionIntegrator(
        ConstantCoefficient( I),
        # Thunderbolt.NodalQuadratureRuleCollection(Thunderbolt.LagrangeCollection{1}()), # Allow e.g. mass lumping for explicit integrators.
        qrc, # Allow e.g. mass lumping for explicit integrators.
        :displacement,
    )
    Thunderbolt.compute_quadrature_fluxes!(∇u_mech, sync.transfer_op_Mech_EP.dh_from, u_mech, :φₘ, integrator)
    fiber_coeff = inner_integrator.f.integrator.volume_model.material_model.microstructure_model.fiber_coefficient
    F = deepcopy(∇u_mech)
    F.data .= Ref(I) .+ F.data
    I4_mech = Thunderbolt.construct_qvector(Vector{Float64}, Vector{Int64}, dh.grid, qrc)
    proj = L2Projector(dh.grid)
    for sdh in dh.subdofhandlers
        fiber_coeff_cache = Thunderbolt.setup_coefficient_cache(fiber_coeff, Thunderbolt.getquadraturerule(qrc,sdh), sdh)
        qr = Thunderbolt.getquadraturerule(qrc,sdh)
        add!(proj, sdh.cellset, Thunderbolt.getinterpolation(LagrangeCollection{1}(), getcells(dh.grid, first(sdh.cellset))) ; qr_rhs = qr)
        for cell in CellIterator(sdh)
            cell_idx = cellid(cell)
            I4_mech_cell = Thunderbolt.get_data_for_index(I4_mech, cell_idx)
            F_cell = Thunderbolt.get_data_for_index(F, cell_idx)
            for qp in QuadratureIterator(qr)
                f = evaluate_coefficient(fiber_coeff_cache, cell, qp, 0.0)
                I4_mech_cell[qp.i] = (F_cell[qp.i] ⋅ f) ⋅ (F_cell[qp.i] ⋅ f)
            end
        end
    end
    I4_vec_of_vecs = Vector{Float64}[]
    for sdh in dh.subdofhandlers
        for cell in CellIterator(sdh)
            cell_idx = cellid(cell)
            I4_mech_cell = Thunderbolt.get_data_for_index(I4_mech, cell_idx)
            push!(I4_vec_of_vecs, I4_mech_cell)
        end
    end
    close!(proj)
    projected = project(proj, I4_vec_of_vecs)
    I4_in_EP = outer_integrator.subintegrator_tree[1][2].f.ode.I4
    Thunderbolt.transfer!(I4_in_EP, sync.transfer_op_Mech_EP, (projected))    
end

function Thunderbolt.semidiscretize(coupled_model::ElectroMechanicalCoupledModel, discretizations::Tuple, meshes::Tuple{<:Thunderbolt.AbstractGrid, <:Thunderbolt.AbstractGrid})
    ep_sdp = semidiscretize(coupled_model.ep_model, discretizations[1], meshes[1])
    mech_sdp = semidiscretize(coupled_model.structural_model, discretizations[2], meshes[2])
    dh_mech_displacement = mech_sdp.dh
    dh_ep_φₘ = ep_sdp.functions[1].dh
    dh_mech_φₘ = create_dh_helper(get_grid(dh_mech_displacement), discretizations[1], :φₘ)
    ep_mech_transfer_op = Thunderbolt.NodalIntergridInterpolation(dh_ep_φₘ, dh_mech_φₘ)
    mech_ep_transfer_op = Thunderbolt.NodalIntergridInterpolation(dh_mech_φₘ, dh_ep_φₘ)
    sync = ElectroMechanicalSynchronizer(ep_mech_transfer_op, mech_ep_transfer_op)
    return GenericSplitFunction(
        (ep_sdp, mech_sdp),
        (Thunderbolt.OS.get_solution_indices(ep_sdp, 2), (last(Thunderbolt.OS.get_solution_indices(ep_sdp, 2))+1):((last(Thunderbolt.OS.get_solution_indices(ep_sdp, 2))) + solution_size(mech_sdp))),
        (OS.NoExternalSynchronization(), sync)
    )
end