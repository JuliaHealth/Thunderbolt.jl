using Test, Thunderbolt, Tensors
using JET: @test_opt
@testset "Element API" begin
    import Thunderbolt: setup_element_cache
    import Thunderbolt: BilinearMassIntegrator, BilinearDiffusionIntegrator
    import FerriteOperators
    import FerriteOperators: setup_facet_item_cache
    import FerriteOperators: assemble_cell!, assemble_facet!, reinit_values!
    import FerriteOperators: CellArgs, FacetArgs, TimeIntegrationContext
    import FerriteOperators:
        ResidualRequest, JacobianRequest, JacobianResidualRequest, WeightedJacobianRequest

    setup_test_cache(kwargs...) =
        FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
    function setup_test_composite_volume_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeVolumetricElementCache((element_cache, element_cache)),
        )
    end
    function setup_test_composite_surface_cache(facets, kwargs...)
        element_cache = FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            setup_facet_item_cache(kwargs...),
        )
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeFacetItemCache(
                (element_cache, element_cache),
                (facets, facets),
            ),
        )
    end

    # A weak boundary condition's facetset IS its traversal, so a hand-driven kernel call walks the
    # facets of that set the cell owns -- which is also the set a `CompositeFacetItemCache` gates
    # each of its inners on.
    declared_facets(cell, name) = Set{FacetIndex}(
        FacetIndex(cellid(cell), lfi) for
        lfi = 1:nfacets(cell) if FacetIndex(cellid(cell), lfi) ∈ getfacetset(cell.grid, name)
    )
    local_facets(cell, name) = sort!([facet[2] for facet in declared_facets(cell, name)])

    grid = generate_grid(Hexahedron, (1, 1, 1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)

    ipv = ip^3
    dhv = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    Ferrite.reinit!(cell_cache_v, 1)
    uₑv =
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ] .* 1e-4

    # The per-sweep scalars every kernel here reads through `evaluation_time`. None of these elements
    # integrates an internal variable over a stage, so the local stage interval is the step size.
    Δt  = 0.25
    ctx = TimeIntegrationContext(0.0, Δt, Δt)

    # No we check some examples for the implemented physics
    @testset "Scalar volumetric bilinear elements: $model" for model in (
        BilinearMassIntegrator(ConstantCoefficient(1.0), qrc, :u),
        BilinearDiffusionIntegrator(ConstantCoefficient(one(Tensor{2, 3})), qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        # A bilinear form's tangent is independent of the unknown, so the sweep declares no slot.
        args = CellArgs((;), cell_cache_s, nothing, ctx)

        element_cache = setup_test_cache(model, sdhs)

        # The engine reinitializes a cache's values objects before the kernel runs; a hand-driven call
        # does the same.
        reinit_values!(element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ¹), element_cache, args)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        reinit_values!(composite_element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Vectorial volumetric nonlinear elements: $model" for model in (
        PK1Model(
            HolzapfelOgden2009Model(),
            ConstantCoefficient(
                OrthotropicMicrostructure(
                    Vec((1.0, 0.0, 0.0)),
                    Vec((0.0, 1.0, 0.0)),
                    Vec((0.0, 0.0, 1.0)),
                ),
            ),
        ),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        args = CellArgs((u = uₑv,), cell_cache_v, nothing, ctx)

        element_cache = setup_test_cache(QuasiStaticModel(:u, model, ()), qr, sdhv)
        reinit_values!(element_cache, cell_cache_v)

        @test_opt assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), element_cache, args)
        assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), element_cache, args)
        @test !iszero(Kₑ¹)
        @test !iszero(rₑ¹)

        @test_opt assemble_cell!(ResidualRequest(rₑ²), element_cache, args)
        assemble_cell!(ResidualRequest(rₑ²), element_cache, args)
        @test rₑ² ≈ rₑ¹

        @test_opt assemble_cell!(JacobianRequest{:u}(Kₑ²), element_cache, args)
        assemble_cell!(JacobianRequest{:u}(Kₑ²), element_cache, args)
        @test Kₑ² ≈ Kₑ¹

        composite_element_cache =
            setup_test_composite_volume_cache(QuasiStaticModel(:u, model, ()), qr, sdhv)
        reinit_values!(composite_element_cache, cell_cache_v)

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        @test_opt assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), composite_element_cache, args)
        assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), composite_element_cache, args)
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        rₑ² .= 0.0
        @test_opt assemble_cell!(ResidualRequest(rₑ²), composite_element_cache, args)
        assemble_cell!(ResidualRequest(rₑ²), composite_element_cache, args)
        @test rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        @test_opt assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        @test Kₑ² ≈ Kₑ¹
    end

    # No we check some examples for the implemented physics
    @testset "Vectorial surface elements: $model" for (model, has_jac) in (
        (RobinBC(1.0, "left"), true),
        (NormalSpringBC(1.0, "left"), true),
        (BendingSpringBC(1.0, "left"), true),
        (ConstantPressureBC(1.0, "left"), true),
        (PressureFieldBC(ConstantCoefficient(1.0), "left"), true),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        args = FacetArgs((u = uₑv,), cell_cache_v, nothing, ctx)

        element_cache = setup_facet_item_cache(model, qrf, sdhv)
        lfis = local_facets(cell_cache_v, "left")

        for local_facet_index in lfis
            assemble_facet!(
                JacobianResidualRequest(Kₑ¹, rₑ¹),
                element_cache,
                args,
                local_facet_index,
            )
            @test iszero(Kₑ¹) != has_jac
            @test iszero(rₑ¹) != has_jac

            assemble_facet!(ResidualRequest(rₑ²), element_cache, args, local_facet_index)
            @test rₑ² ≈ rₑ¹

            assemble_facet!(JacobianRequest{:u}(Kₑ²), element_cache, args, local_facet_index)
            @test Kₑ² ≈ Kₑ¹
        end

        composite_element_cache = setup_test_composite_surface_cache(
            declared_facets(cell_cache_v, "left"),
            model,
            qrf,
            sdhv,
        )

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        for local_facet_index in lfis
            assemble_facet!(
                JacobianResidualRequest(Kₑ¹, rₑ¹),
                composite_element_cache,
                args,
                local_facet_index,
            )
        end
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        rₑ² .= 0.0
        for local_facet_index in lfis
            assemble_facet!(ResidualRequest(rₑ²), composite_element_cache, args, local_facet_index)
            assemble_facet!(
                JacobianRequest{:u}(Kₑ²),
                composite_element_cache,
                args,
                local_facet_index,
            )
        end
        @test Kₑ² ≈ Kₑ¹
        @test rₑ² ≈ rₑ¹
    end

    # The dashpots are functions of `(u, v)` rather than `(u, t)`, and `:v` is a reconstructed slot:
    # the engine forms it as `∂v∂u * (u - uᵥ)` at gather time, so a hand-built `args` supplies the
    # already-gathered cell-local value. `∂v∂u` is the scheme's reconstruction slope, `1/Δt` for
    # backward Euler.
    @testset "Viscous surface elements: $model" for model in (
        ViscousRobinBC(3.0, "left"),
        ViscousNormalSpringBC(3.0, "left"),
    )
        n = ndofs(dhv)
        uprev = uₑv ./ 3
        ∂v∂u = inv(Δt)

        cache = setup_facet_item_cache(model, qrf, sdhv)
        lfis = local_facets(cell_cache_v, "left")

        rate_args(slope) = FacetArgs(
            (u = uₑv, v = slope .* (uₑv .- uprev)),
            cell_cache_v,
            nothing,
            TimeIntegrationContext(0.0, inv(slope), inv(slope)),
        )
        args = rate_args(∂v∂u)

        K¹ = zeros(n, n)
        r¹ = zeros(n)
        for lfi in lfis
            assemble_facet!(JacobianResidualRequest(K¹, r¹), cache, args, lfi)
        end
        @test !iszero(r¹)
        # `:v` is frozen under a `:u` Jacobian, so a dashpot contributes nothing to ∂F/∂u.
        @test iszero(K¹)

        # The three assembly variants must agree.
        K² = zeros(n, n)
        r² = zeros(n)
        for lfi in lfis
            assemble_facet!(ResidualRequest(r²), cache, args, lfi)
            assemble_facet!(JacobianRequest{:u}(K²), cache, args, lfi)
        end
        @test r² ≈ r¹
        @test K² ≈ K¹

        # The damping block `∂v∂u ⋅ D` is served by the weighted request, which carries the scheme's
        # chain-rule scalars as request payload.
        Kv = zeros(n, n)
        for lfi in lfis
            assemble_facet!(WeightedJacobianRequest(Kv, (u = 1.0, v = ∂v∂u)), cache, args, lfi)
        end
        @test !iszero(Kv)

        # The family is linear in the velocity, so the residual is *exactly* the damping block applied
        # to the increment -- an identity, not an approximation. It subsumes three weaker checks at
        # once: that the block is the derivative of the residual, that no velocity means no traction,
        # and that the block does not depend on `u`.
        @test r¹ ≈ Kv * (uₑv .- uprev)

        # Halving the timestep doubles the reconstruction slope, hence both the traction and the
        # damping block.
        Kh = zeros(n, n)
        rh = zeros(n)
        args_h = rate_args(2∂v∂u)
        for lfi in lfis
            assemble_facet!(ResidualRequest(rh), cache, args_h, lfi)
            assemble_facet!(WeightedJacobianRequest(Kh, (u = 1.0, v = 2∂v∂u)), cache, args_h, lfi)
        end
        @test Kh ≈ 2 .* Kv
        @test rh ≈ 2 .* r¹

        # The traction is linear in the viscosity, which is the whole of `damping_tensor`.
        stiffer = setup_facet_item_cache(
            model isa ViscousRobinBC ? ViscousRobinBC(6.0, "left") :
            ViscousNormalSpringBC(6.0, "left"),
            qrf,
            sdhv,
        )
        K2 = zeros(n, n)
        r2 = zeros(n)
        for lfi in lfis
            assemble_facet!(ResidualRequest(r2), stiffer, args, lfi)
            assemble_facet!(WeightedJacobianRequest(K2, (u = 1.0, v = ∂v∂u)), stiffer, args, lfi)
        end
        @test K2 ≈ 2 .* Kv
        @test r2 ≈ 2 .* r¹

        # The constitutive difference between the two dashpots, and the only place it is visible: the
        # normal one leaves tangential sliding free. Facet "left" has n₀ = -e₁, so a velocity along e₂
        # is purely tangential and must draw no traction from it at all. Exact, so no tolerance.
        vtan = zeros(n)
        for k = 1:8
            vtan[3*(k-1)+2] = 1.0e-3
        end
        rtan = zeros(n)
        tan_args = FacetArgs((u = vtan, v = ∂v∂u .* vtan), cell_cache_v, nothing, ctx)
        for lfi in lfis
            assemble_facet!(ResidualRequest(rtan), cache, tan_args, lfi)
        end
        @test iszero(rtan) == (model isa ViscousNormalSpringBC)
    end

    @testset "Mixed spring/dashpot boundary" begin
        # Two terms supported on the SAME surface, which is what the facet-item composite's union of
        # declarations makes legal: one item, both inners assembling it. One `FacetArgs` serves a
        # composite whose inners read *different* slots -- the spring reads the trial displacement
        # `:u`, the dashpot the reconstructed rate `:v` -- and each takes the slot it needs from it.
        n     = ndofs(dhv)
        uprev = uₑv ./ 3
        ∂v∂u  = inv(Δt)
        args  = FacetArgs((u = uₑv, v = ∂v∂u .* (uₑv .- uprev)), cell_cache_v, nothing, ctx)

        spring = setup_facet_item_cache(NormalSpringBC(5.0, "left"), qrf, sdhv)
        dashpot = setup_facet_item_cache(ViscousRobinBC(3.0, "left"), qrf, sdhv)
        facets = declared_facets(cell_cache_v, "left")
        composite = FerriteOperators.CompositeFacetItemCache((spring, dashpot), (facets, facets))
        lfis = local_facets(cell_cache_v, "left")

        rs = zeros(n);
        rd = zeros(n);
        rc = zeros(n)
        for lfi in lfis
            assemble_facet!(ResidualRequest(rs), spring, args, lfi)
            assemble_facet!(ResidualRequest(rd), dashpot, args, lfi)
            assemble_facet!(ResidualRequest(rc), composite, args, lfi)
        end
        @test !iszero(rs)
        @test !iszero(rd)
        @test rc ≈ rs .+ rd

        # The weighted request is the one carrying both blocks -- the spring's ∂F/∂u and the dashpot's
        # `∂v∂u ⋅ D` -- so it is where the composite's tangent is the sum of its inners'.
        weights = (u = 1.0, v = ∂v∂u)
        Ks = zeros(n, n);
        Kd = zeros(n, n);
        Kc = zeros(n, n)
        for lfi in lfis
            assemble_facet!(WeightedJacobianRequest(Ks, weights), spring, args, lfi)
            assemble_facet!(WeightedJacobianRequest(Kd, weights), dashpot, args, lfi)
            assemble_facet!(WeightedJacobianRequest(Kc, weights), composite, args, lfi)
        end
        @test !iszero(Kd)
        @test Kc ≈ Ks .+ Kd
    end
end

@testset "Collocated nodal quadrature (spectral-element mass)" begin
    import Thunderbolt:
        NodalQuadratureRuleCollection,
        LagrangeCollection,
        DiscontinuousLagrangeCollection,
        getquadraturerule
    import Ferrite
    using LinearAlgebra: Diagonal, diag

    # The mass a collocated rule assembles over a tensor-product Lagrange space is diagonal, and its
    # diagonal sums to the domain measure -- the property the whole spectral-element arm rests on.
    function nodal_mass(celltype, refshape, order, dims, extent)
        grid = Ferrite.generate_grid(celltype, dims, zero(extent), extent)
        dh = Ferrite.DofHandler(grid)
        Ferrite.add!(dh, :u, Ferrite.Lagrange{refshape, order}())
        Ferrite.close!(dh)
        sdh = dh.subdofhandlers[1]
        qr = getquadraturerule(NodalQuadratureRuleCollection(LagrangeCollection{order}()), sdh)
        cv = Ferrite.CellValues(
            qr,
            Ferrite.Lagrange{refshape, order}(),
            Ferrite.geometric_interpolation(celltype),
        )
        M = Ferrite.allocate_matrix(dh)
        asm = Ferrite.start_assemble(M)
        n = Ferrite.getnbasefunctions(cv)
        Me = zeros(n, n)
        for cell in Ferrite.CellIterator(dh)
            Ferrite.reinit!(cv, cell)
            fill!(Me, 0)
            for qp = 1:Ferrite.getnquadpoints(cv), i = 1:n, j = 1:n
                Me[i, j] +=
                    Ferrite.shape_value(cv, qp, i) *
                    Ferrite.shape_value(cv, qp, j) *
                    Ferrite.getdetJdV(cv, qp)
            end
            Ferrite.assemble!(asm, Ferrite.celldofs(cell), Me)
        end
        return M
    end

    @testset "$name" for (name, celltype, refshape, dims, extent, measure) in (
            ("quad", Quadrilateral, Ferrite.RefQuadrilateral, (2, 3), Vec{2}((1.5, 0.5)), 0.75),
            ("hex", Hexahedron, Ferrite.RefHexahedron, (2, 2, 2), Vec{3}((1.3, 1.3, 1.3)), 1.3^3),
        ),
        order in (1, 2)

        M = nodal_mass(celltype, refshape, order, dims, extent)
        d = diag(M)
        @test minimum(d) > 0
        @test maximum(abs, M - Diagonal(d)) < 1.0e-12 * minimum(d)
        @test sum(d) ≈ measure rtol = 1.0e-12
    end

    # Both Lagrange families share the node set, so both get real weights; a simplex has no such
    # collocated rule and keeps the documented NaN.
    let cellhex = Ferrite.getcells(Ferrite.generate_grid(Hexahedron, (1, 1, 1)), 1),
        celltet = Ferrite.getcells(Ferrite.generate_grid(Tetrahedron, (1, 1, 1)), 1)

        for ipc in (LagrangeCollection{2}(), DiscontinuousLagrangeCollection{2}())
            qr = getquadraturerule(NodalQuadratureRuleCollection(ipc), cellhex)
            @test all(isfinite, Ferrite.getweights(qr))
            @test sum(Ferrite.getweights(qr)) ≈ 8.0   # the reference hexahedron's measure
        end
        qr = getquadraturerule(NodalQuadratureRuleCollection(LagrangeCollection{1}()), celltet)
        @test all(isnan, Ferrite.getweights(qr))
    end
end

@testset "Mass treatments of the discretization" begin
    using LinearAlgebra: Diagonal, diag
    mesh  = generate_mesh(Quadrilateral, (3, 2), Vec{2}((0.0, 0.0)), Vec{2}((1.5, 0.5)))
    model = TransientDiffusionModel(ConstantCoefficient(one(Tensor{2, 2})), NoStimulationProtocol(), :u)
    ctx   = TimeIntegrationContext(0.0, 0.0, 0.0)
    function mass_matrix(order; kwargs...)
        f  = semidiscretize(model, FiniteElementDiscretization(Dict(:u => LagrangeCollection{order}()); kwargs...), mesh)
        op = FerriteOperators.setup_operator(FerriteOperators.AssemblyStrategy(FerriteOperators.SequentialCPUDevice()), f.mass_term, f.dh)
        FerriteOperators.update_operator!(op, nothing, ctx)
        return op, FerriteOperators.get_matrix(op)
    end

    _, M = mass_matrix(2)
    lop, D = mass_matrix(2; mass = LumpedMass())
    @test D isa Diagonal
    @test diag(D) ≈ vec(sum(M, dims = 2))

    # The collocated element writes the diagonal the nodal rule's consistent assembly would produce.
    _, Mnodal = mass_matrix(
        2;
        qrcs = Dict{Symbol, Any}(:mass => NodalQuadratureRuleCollection(LagrangeCollection{2}())),
    )
    _, C = mass_matrix(2; mass = CollocatedMass())
    @test C isa Diagonal
    @test diag(C) ≈ diag(Mnodal)
    @test maximum(abs, Mnodal - C) < 1.0e-12 * minimum(diag(C))
    @test sum(diag(C)) ≈ 0.75 rtol = 1.0e-12

    @test_throws ErrorException mass_matrix(
        2;
        mass = CollocatedMass(),
        qrcs = Dict{Symbol, Any}(:mass => NodalQuadratureRuleCollection(LagrangeCollection{2}())),
    )
    # The implicit solvers combine M and K on one pattern, which a `Diagonal` has not.
    @test_throws ErrorException Thunderbolt._assert_combinable_mass(lop)
end
