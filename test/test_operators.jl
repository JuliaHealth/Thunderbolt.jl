using Thunderbolt
import Thunderbolt: AssembledNonlinearOperator, AssembledBilinearOperator, NullOperator, DiagonalOperator, BlockOperator, EAVector
import LinearAlgebra: mul!
using BlockArrays, SparseArrays, StaticArrays, Test

@testset "Operators" begin
    @testset "Actions" begin
        vin = ones(5)
        vout = ones(5)

        nullop = NullOperator{Float64,5,5}()
        @test eltype(nullop) == Float64
        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)

        mul!(vout, nullop, vin)
        @test vout == zeros(5)

        vout .= ones(5)
        mul!(vout, nullop, vin, 2.0, 1.0)
        @test vout == ones(5)

        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)
        
        @test Thunderbolt.getJ(nullop) ≈ zeros(5,5)


        diagop = DiagonalOperator([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        mul!(vout, diagop, vin)
        @test vout == [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, 1.0, 1.0)
        @test vout == 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, -2.0, 1.0)
        @test vout == zeros(5)
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        
        @test Thunderbolt.getJ(diagop) ≈ spdiagm([1.0, 2.0, 3.0, 4.0, 5.0])


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64,4,5}()

        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)
        @test vout == vout
        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)

        @test Thunderbolt.getJ(nullop_rect) ≈ zeros(4,5)


        vin = mortar([ones(4), -ones(2)])
        vout = mortar([-ones(4), -ones(2)])
        bop_id = BlockOperator((
            DiagonalOperator(ones(4)), NullOperator{Float64,2,4}(),
            NullOperator{Float64,4,2}(),  DiagonalOperator(ones(2))
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
        grid = generate_grid(Quadrilateral, (2,2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}())
        close!(dh)
        qrc = QuadratureRuleCollection{2}()

        @testset "Constant Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> 1.0, cs),
                [SVector((0.0, 1.0))]
            )

            for strategy_cache in (
                    Thunderbolt.SequentialAssemblyStrategyCache(nothing),
                    Thunderbolt.ElementAssemblyStrategyCache(SequentialCPUDevice(),EAVector(dh)),
                    Thunderbolt.ElementAssemblyStrategyCache(PolyesterDevice(),EAVector(dh)),
            )
                linop = Thunderbolt.LinearOperator(
                    zeros(ndofs(dh)),
                    protocol,
                    qrc,
                    dh,
                    strategy_cache,
                )
                Thunderbolt.update_operator!(linop,0.0)
                @test linop.b ≈ [0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.5, 0.25, 0.25]
            end
        end

        @testset "Quadratic Cartesian" begin
            cs = CartesianCoordinateSystem(grid)
            protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> norm(x)^2+1.0, cs),
                [SVector((0.0, 1.0))]
            )

            for strategy_cache in (
                Thunderbolt.SequentialAssemblyStrategyCache(nothing),
                Thunderbolt.ElementAssemblyStrategyCache(SequentialCPUDevice(),EAVector(dh)),
                Thunderbolt.ElementAssemblyStrategyCache(PolyesterDevice(),EAVector(dh)),
            )
                linop = Thunderbolt.LinearOperator(
                    zeros(ndofs(dh)),
                    protocol,
                    qrc,
                    dh,
                    strategy_cache,
                )
                Thunderbolt.update_operator!(linop,0.0)
                @test linop.b ≈ [1.0/2, 5.0/6, 4.0/3, 5.0/6, 1.0/2, 5.0/6, 5.0/6, 1.0/2, 1.0/2]
            end
        end
    end

    @testset "Coupled" begin
        # TODO test with faulty blocks
        @testset "Block action" begin
            op1 = BlockOperator((
                DiagonalOperator([1.0, 2.0, 3.0]), NullOperator{Float64,2,3}(),
                NullOperator{Float64,3,2}(), NullOperator{Float64,2,2}()
            ))
            @test op1 * [1.0, 2.0, 3.0, 4.0, 5.0] ≈ [1.0, 4.0, 9.0, 0.0, 0.0]

            op2 = BlockOperator((
                NullOperator{Float64,3,3}(), NullOperator{Float64,2,3}(),
                NullOperator{Float64,3,2}(), DiagonalOperator([-1.0, 2.0])
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
