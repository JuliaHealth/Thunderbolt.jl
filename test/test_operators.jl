using Thunderbolt
import Thunderbolt:
    NullOperator,
    DiagonalOperator,
    BlockOperator,
    EAVector,
    BilinearDiffusionIntegrator,
    NonlinearIntegrator,
    FacetQuadratureRuleCollection,
    InternalVariableHandler,
    QuasiStaticFunction
import LinearAlgebra: mul!
using BlockArrays, SparseArrays, StaticArrays, Test

@testset "Operators" begin
    solver = BackwardEulerSolver() # TODO remove this

    @testset "Actions" begin
        vin = ones(5)
        vout = ones(5)

        nullop = NullOperator{Float64, 5, 5}()
        @test eltype(nullop) == Float64
        @test length(vin) == size(nullop, 1)
        @test length(vout) == size(nullop, 2)

        mul!(vout, nullop, vin)
        @test vout == zeros(5)

        vout .= ones(5)
        mul!(vout, nullop, vin, 2.0, 1.0)
        @test vout == ones(5)

        @test length(vin) == size(nullop, 1)
        @test length(vout) == size(nullop, 2)

        @test Thunderbolt.getJ(nullop) ≈ zeros(5, 5)


        diagop = DiagonalOperator([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(vin) == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        mul!(vout, diagop, vin)
        @test vout == [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, 1.0, 1.0)
        @test vout == 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, -2.0, 1.0)
        @test vout == zeros(5)
        @test length(vin) == size(diagop, 1)
        @test length(vout) == size(diagop, 2)

        @test Thunderbolt.getJ(diagop) ≈ spdiagm([1.0, 2.0, 3.0, 4.0, 5.0])


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64, 4, 5}()

        @test length(vin) == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)
        @test vout == vout
        @test length(vin) == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)

        @test Thunderbolt.getJ(nullop_rect) ≈ zeros(4, 5)


        vin = mortar([ones(4), -ones(2)])
        vout = mortar([-ones(4), -ones(2)])
        bop_id = BlockOperator((
            DiagonalOperator(ones(4)),
            NullOperator{Float64, 2, 4}(),
            NullOperator{Float64, 4, 2}(),
            DiagonalOperator(ones(2)),
        ))

        @test vout != vin
        mul!(vout, bop_id, vin)
        @test vout == vin
        mul!(vout, bop_id, vin, 2.0, 1.0)
        @test vout == 3.0*vin

        @test Thunderbolt.getJ(bop_id) ≈ spdiagm(ones(6))
    end

    @testset "Linear" begin
        # Setup
        grid = generate_grid(Quadrilateral, (4, 3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        qrc = QuadratureRuleCollection{2}()

        @testset "Constant Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            linint = Thunderbolt.LinearIntegrator(
                AnalyticalTransmembraneStimulationProtocol(
                    AnalyticalCoefficient((x, t) -> 1.0, cs),
                    [SVector((0.0, 1.0))],
                ),
                qrc,
            )

            linop_base = Thunderbolt.setup_operator(
                Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
                linint,
                solver,
                dh,
            )
            # Check that assembly works
            Thunderbolt.update_operator!(linop_base, 0.0)
            norm_baseline = norm(linop_base.b)
            @test norm_baseline > 0.0
            # Idempotency
            Thunderbolt.update_operator!(linop_base, 0.0)
            @test norm_baseline == norm(linop_base.b)

            @testset "Strategy $strategy" for strategy in (
                Thunderbolt.ElementAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(3)),
                Thunderbolt.PerColorAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
            )
                linop = Thunderbolt.setup_operator(strategy, linint, solver, dh)

                # Consistency
                Thunderbolt.update_operator!(linop, 0.0)
                @test linop.b ≈ linop_base.b
                # Idempotency
                Thunderbolt.update_operator!(linop, 0.0)
                @test linop.b ≈ linop_base.b
            end
        end

        @testset "Quadratic Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            linint = Thunderbolt.LinearIntegrator(
                AnalyticalTransmembraneStimulationProtocol(
                    AnalyticalCoefficient((x, t) -> norm(x)^2+1.0, cs),
                    [SVector((0.0, 1.0))],
                ),
                qrc,
            )

            linop_base = Thunderbolt.setup_operator(
                Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
                linint,
                solver,
                dh,
            )

            # Check that assembly works
            Thunderbolt.update_operator!(linop_base, 0.0)
            norm_baseline = norm(linop_base.b)
            @test norm_baseline > 0.0
            # Idempotency
            Thunderbolt.update_operator!(linop_base, 0.0)
            @test norm_baseline == norm(linop_base.b)

            @testset "Strategy $strategy" for strategy in (
                Thunderbolt.ElementAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.ElementAssemblyStrategy(PolyesterDevice(3)),
                Thunderbolt.PerColorAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
            )
                linop = Thunderbolt.setup_operator(strategy, linint, solver, dh)

                # Check that assembly works
                Thunderbolt.update_operator!(linop_base, 0.0)
                norm_baseline = norm(linop_base.b)
                @test norm_baseline > 0.0
                # Idempotency
                Thunderbolt.update_operator!(linop_base, 0.0)
                @test norm_baseline == norm(linop_base.b)
            end
        end
    end

    @testset "Bilinear" begin
        # Setup
        grid = generate_grid(Quadrilateral, (10, 9))
        Ferrite.transform_coordinates!(grid, x->Vec{2}(sign.(x .- 0.5) .* (x .- 0.5) .^ 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        qrc = QuadratureRuleCollection{2}()

        @testset "Constant Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            integrator = BilinearDiffusionIntegrator(
                ConstantCoefficient(SymmetricTensor{2, 2, Float64, 3}((4.5e-5, 0, 2.0e-5))),
                QuadratureRuleCollection(2),
                :u,
            )
            bilinop_base = Thunderbolt.setup_operator(
                Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
                integrator,
                solver,
                dh,
            )
            # Check that assembly works
            Thunderbolt.update_operator!(bilinop_base, 0.0)
            norm_baseline = norm(bilinop_base.A)
            @test norm_baseline > 0.0
            # Idempotency
            Thunderbolt.update_operator!(bilinop_base, 0.0)
            @test norm_baseline == norm(bilinop_base.A)

            @testset "Strategy $strategy" for strategy in (
                Thunderbolt.PerColorAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
            )
                bilinop = Thunderbolt.setup_operator(strategy, integrator, solver, dh)
                # Consistency
                Thunderbolt.update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
                # Idempotency
                Thunderbolt.update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
            end
        end

        @testset "Analytical coefficient LVCS" begin
            cs = LVCoordinateSystem(
                dh,
                LagrangeCollection{1}(),
                rand(ndofs(dh)),
                rand(ndofs(dh)),
                rand(ndofs(dh)),
            )
            integrator = BilinearDiffusionIntegrator(
                AnalyticalCoefficient(
                    (x, t) ->
                        SymmetricTensor{2, 2, Float64, 3}((abs(x.transmural)+1e-6, 0, 2.0e-5)),
                    cs,
                ),
                QuadratureRuleCollection(2),
                :u,
            )
            bilinop_base = Thunderbolt.setup_operator(
                Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
                integrator,
                solver,
                dh,
            )
            # Check that assembly works
            Thunderbolt.update_operator!(bilinop_base, 0.0)
            norm_baseline = norm(bilinop_base.A)
            @test norm_baseline > 0.0
            # Idempotency
            Thunderbolt.update_operator!(bilinop_base, 0.0)
            @test norm_baseline == norm(bilinop_base.A)

            @testset "Strategy $strategy" for strategy in (
                Thunderbolt.PerColorAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
            )
                bilinop = Thunderbolt.setup_operator(strategy, integrator, solver, dh)
                # Consistency
                Thunderbolt.update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
                # Idempotency
                Thunderbolt.update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
            end
        end
    end

    @testset "Nonlinear" begin
        # TODO remove
        solver = NewtonRaphsonSolver()

        # Setup
        grid = generate_grid(Hexahedron, (2, 5, 9))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x .- 0.5) .* (x .- 0.5) .^ 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dh)
        qrc = QuadratureRuleCollection{2}()
        lvh = InternalVariableHandler(grid)
        # close!(lvh)
        ch = ConstraintHandler(dh)
        close!(ch)

        @testset "Constant Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            integrator = NonlinearIntegrator(
                QuasiStaticModel(
                    :u,
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
                    (),
                ),
                (),
                [:u],
                QuadratureRuleCollection(2),
                FacetQuadratureRuleCollection(2),
            )
            u = zeros(ndofs(dh))
            apply_analytical!(u, dh, :u, x -> 0.05x)
            nlop_base = Thunderbolt.setup_operator(
                QuasiStaticFunction(
                    dh,
                    ch,
                    lvh,
                    integrator,
                    Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
                ),
                solver,
            )
            # Check that assembly works
            residual_base = zeros(ndofs(dh))
            Thunderbolt.update_linearization!(nlop_base, residual_base, u, 0.0)
            norm_baseline = norm(nlop_base.J)
            @test norm_baseline > 0.0
            # Idempotency
            Thunderbolt.update_linearization!(nlop_base, u, 0.0)
            @test norm_baseline == norm(nlop_base.J)

            residual = zeros(ndofs(dh))
            @testset "Strategy $strategy" for strategy in (
                Thunderbolt.PerColorAssemblyStrategy(SequentialCPUDevice()),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(1)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(2)),
                Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
            )
                nlop = Thunderbolt.setup_operator(
                    QuasiStaticFunction(dh, ch, lvh, integrator, strategy),
                    solver,
                )
                # Consistency
                Thunderbolt.update_linearization!(nlop, residual, u, 0.0)
                @test residual ≈ residual_base
                @test nlop.J ≈ nlop_base.J
                Thunderbolt.update_linearization!(nlop, u, 0.0)
                @test nlop.J ≈ nlop_base.J
                # Idempotency
                Thunderbolt.update_linearization!(nlop, residual, u, 0.0)
                @test residual ≈ residual_base
                @test nlop.J ≈ nlop_base.J
                Thunderbolt.update_linearization!(nlop, u, 0.0)
                @test nlop.J ≈ nlop_base.J
            end
        end
    end

    @testset "Coupled" begin
        # TODO test with faulty blocks
        @testset "Block action" begin
            op1 = BlockOperator((
                DiagonalOperator([1.0, 2.0, 3.0]),
                NullOperator{Float64, 2, 3}(),
                NullOperator{Float64, 3, 2}(),
                NullOperator{Float64, 2, 2}(),
            ))
            @test op1 * [1.0, 2.0, 3.0, 4.0, 5.0] ≈ [1.0, 4.0, 9.0, 0.0, 0.0]

            op2 = BlockOperator((
                NullOperator{Float64, 3, 3}(),
                NullOperator{Float64, 2, 3}(),
                NullOperator{Float64, 3, 2}(),
                DiagonalOperator([-1.0, 2.0]),
            ))
            @test op2 * [1.0, 2.0, 3.0, 4.0, 5.0] ≈ [0.0, 0.0, 0.0, -4.0, 10.0]
        end

        @testset "Block elimination" begin
            # Idea
            # 1. Setup coupled problem with diffusion diffusion coupling as in Bidomain model
            # 2. Setup equivalent Bidomain system matrix manually
            # 3. Check that problem is singular in both cases
            # 4. Eliminate system from 2. manually
            # 5. Eliminate system from 3. with Thunderbolt.jl
            # 6. Check elimination gives the same result
            # 7. Check that problem is not singular anymore
            @test_broken false && "Implement me!"
        end
    end
end
