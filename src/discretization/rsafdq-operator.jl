# TODO try to reproduce this via the BlockOperator
struct AssembledRSAFDQ2022Operator{MatrixType <: BlockMatrix, ElementModelType, FacetModelType, TyingModelType, DHType} <: AbstractBlockOperator
    J::MatrixType
    element_model::ElementModelType
    element_qrc::Union{<:QuadratureRuleCollection, Nothing}
    boundary_model::FacetModelType
    boundary_qrc::Union{<:FacetQuadratureRuleCollection, Nothing}
    tying_model::TyingModelType
    tying_qrc::Union{<:QuadratureRuleCollection, <: FacetQuadratureRuleCollection, Nothing}
    dh::DHType
end

function AssembledRSAFDQ2022Operator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FacetQuadratureRuleCollection, tying::RSAFDQ2022TyingInfo)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    Jmech = allocate_matrix(dh)

    num_chambers = length(tying.chambers)
    block_sizes = [ndofs(dh), num_chambers]
    total_size = sum(block_sizes)
    # First we initialize an empty dummy block array
    Jblock = BlockArray(spzeros(total_size,total_size), block_sizes, block_sizes)
    Jblock[Block(1,1)] = Jmech

    AssembledRSAFDQ2022Operator(
        Jblock,
        element_model,
        element_qrc,
        boundary_model,
        boundary_qrc,
        tying,
        boundary_qrc,
        dh,
    )
end

getJ(op::AssembledRSAFDQ2022Operator) = op.J
getJ(op::AssembledRSAFDQ2022Operator, i::Block) = @view op.J[i]

function update_linearization!(op::AssembledRSAFDQ2022Operator, u_::AbstractVector, time)
    @unpack J, dh  = op
    @unpack element_model, element_qrc = op
    @unpack boundary_model, boundary_qrc = op
    @unpack tying_model, tying_qrc = op

    bs = blocksizes(J)
    s1 = bs[1,1][1]
    s2 = bs[2,2][1]
    u  = BlockedVector(u_, [s1, s2])
    ud = @view u[Block(1)]
    up = @view u[Block(2)]

    Jdd = @view J[Block(1,1)]
    Jpd = @view J[Block(2,1)]
    Jdp = @view J[Block(1,2)]

    # Reset residual and Jacobian to 0
    assembler = start_assemble(Jdd)
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

    @assert length(dh.subdofhandlers) == 1
    sdh = first(dh.subdofhandlers)
    #for sdh in dh.subdofhandlers
        element_qr  = getquadraturerule(element_qrc, sdh)
        boundary_qr = getquadraturerule(boundary_qrc, sdh)
        tying_qr    = getquadraturerule(tying_qrc, sdh)

        element_cache  = setup_element_cache(element_model, element_qr, sdh)
        boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, sdh)
        tying_cache    = setup_tying_cache(tying_model, tying_qr, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_J!(assembler, sdh, element_cache, boundary_cache, tying_cache, u, time)

        # Assemble forward and backward coupling contributions
        for (chamber_index,chamber) ∈ enumerate(tying_cache.chambers)
            V⁰ᴰ = chamber.V⁰ᴰval
            chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

            Jpd_current = @view Jpd[chamber_index,:]
            Jdp_current = @view Jdp[:,chamber_index]

            # We cannot update the residual for the displacement block here, because it would be assembled essentially twice.
            @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, dh, ud, chamber_pressure, chamber)
            @timeit_debug "assemble backward coupler" assemble_LFSI_coupling_contribution_row!(Jpd_current, dh, ud, chamber_pressure, V⁰ᴰ, chamber)

            @info chamber_index, chamber_pressure, V⁰ᴰ
        end
    #end

    #finish_assemble(assembler)
end

function update_linearization!(op::AssembledRSAFDQ2022Operator, residual_::AbstractVector, u_::AbstractVector, time)
    @unpack J, dh  = op
    @unpack element_model, element_qrc = op
    @unpack boundary_model, boundary_qrc = op
    @unpack tying_model, tying_qrc = op

    @assert length(dh.field_names) == 1 "Please use block operators for problems with multiple fields."
    field_name = first(dh.field_names)

    bs = blocksizes(J)
    s1 = bs[1,1][1]
    s2 = bs[2,2][1]
    u  = BlockedVector(u_, [s1, s2])
    ud = @view u[Block(1)]
    up = @view u[Block(2)]

    residual  = BlockedVector(residual_, [s1, s2])
    residuald = @view residual[Block(1)]
    residualp = @view residual[Block(2)]

    Jdd = @view J[Block(1,1)]
    Jpd = @view J[Block(2,1)]
    Jdp = @view J[Block(1,2)]
    # Reset residual and Jacobian to 0
    #assembler = start_assemble(Jdd, residuald) # Does not work yet
    assembler = start_assemble(Jdd, residual_) # FIXME
    fill!(residuald, 0.0)
    fill!(residualp, 0.0)
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

    @assert length(dh.subdofhandlers) == 1
    sdh = first(dh.subdofhandlers)
    #for sdh in dh.subdofhandlers
        element_qr  = getquadraturerule(element_qrc, sdh)
        boundary_qr = getquadraturerule(boundary_qrc, sdh)
        tying_qr    = getquadraturerule(tying_qrc, sdh)

        element_cache  = setup_element_cache(element_model, element_qr, sdh)
        boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, sdh)
        tying_cache    = setup_tying_cache(tying_model, tying_qr, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, boundary_cache, tying_cache, u, time)

        # Assemble forward and backward coupling contributions
        for (chamber_index,chamber) ∈ enumerate(tying_cache.chambers)
            V⁰ᴰ = chamber.V⁰ᴰval
            chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

            Jpd_current = @view Jpd[chamber_index,:]
            Jdp_current = @view Jdp[:,chamber_index]

            residualp_current = @view residualp[chamber_index]

            # We cannot update the residual for the displacement block here, because it would be assembled essentially twice.
            @timeit_debug "assemble forward coupler" assemble_LFSI_coupling_contribution_col!(Jdp_current, dh, ud, chamber_pressure, chamber)
            @timeit_debug "assemble backward coupler" assemble_LFSI_coupling_contribution_row!(Jpd_current, residualp_current, dh, ud, chamber_pressure, V⁰ᴰ, chamber)

            @info "Chamber $chamber_index p=$chamber_pressure, V0=$V⁰ᴰ"
        end
    #end

    #finish_assemble(assembler)
end

function setup_operator(f::RSAFDQ20223DFunction, solver::AbstractNonlinearSolver)
    @unpack tying_info, structural_function = f
    @unpack dh, integrator = structural_function
    @unpack fqrc, qrc, volume_model, facet_model = integrator
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    displacement_symbol = first(dh.field_names)

    return AssembledRSAFDQ2022Operator(
        dh, displacement_symbol, volume_model, qrc, facet_model, fqrc, tying_info
    )
end


BlockArrays.blocks(f::RSAFDQ20223DFunction) = (f.structural_function, f.tying_info)
