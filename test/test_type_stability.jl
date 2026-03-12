@testset "Type Stability" begin
    f₀       = Tensors.Vec{3, Float64}((1.0, 0.0, 0.0))
    s₀       = Tensors.Vec{3, Float64}((0.0, 1.0, 0.0))
    n₀       = Tensors.Vec{3, Float64}((0.0, 0.0, 1.0))
    fsncoeff = ConstantCoefficient(OrthotropicMicrostructure(f₀, s₀, n₀))
    fsneval  = Thunderbolt.OrthotropicMicrostructure(f₀, s₀, n₀)
    F        = one(Tensors.Tensor{2, 3})
    Caᵢ      = 0.0

    material_model_set = [
        NullEnergyModel(),
        HolzapfelOgden2009Model(),
        TransverseIsotopicNeoHookeanModel(),
        LinYinPassiveModel(),
        LinYinActiveModel(),
        HumphreyStrumpfYinModel()
    ]

    compression_model_set = [
        NullCompressionPenalty(),
        HartmannNeffCompressionPenalty1(),
        HartmannNeffCompressionPenalty2(),
        HartmannNeffCompressionPenalty3(),
        SimpleCompressionPenalty()
    ]

    @testset "Energies $material_model" for material_model in material_model_set
        @test_opt Thunderbolt.Ψ(F, fsneval, material_model)
        @test Thunderbolt.Ψ(F, fsneval, material_model) == 0.0
    end

    @testset "Compression $compression_model" for compression_model in compression_model_set
        @test_opt Thunderbolt.U(-eps(Float64), compression_model)
        @test_opt Thunderbolt.U(0.0, compression_model)
        @test_opt Thunderbolt.U(1.0, compression_model)
        @test Thunderbolt.U(1.0, compression_model) == 0.0
    end

    @testset "Constitutive Models" begin
        passive_spring = HolzapfelOgden2009Model()

        @testset failfast=true "PK1 Stress" begin
            model = PK1Model(passive_spring, fsncoeff)
            @test_opt Thunderbolt.stress_function(
                model,
                F,
                fsneval,
                Thunderbolt.EmptyInternalModel()
            )
            @test Thunderbolt.stress_function(
                model, F, fsneval, Thunderbolt.EmptyInternalModel())≈
            zero(Tensor{2, 3}) atol=1e-16
            @test_opt Thunderbolt.stress_and_tangent(
                model,
                F,
                fsneval,
                Thunderbolt.EmptyInternalModel()
            )
            P, 𝔸 = Thunderbolt.stress_and_tangent(
                model, F, fsneval, Thunderbolt.EmptyInternalModel())
            @test P≈zero(Tensor{2, 3}) atol=1e-16
        end

        active_stress_set = [SimpleActiveStress(), PiersantiActiveStress()]
        contraction_model_set = [ConstantStretchModel(), PelceSunLangeveld1995Model()]
        Fᵃmodel_set = [
            GMKActiveDeformationGradientModel(),
            GMKIncompressibleActiveDeformationGradientModel(),
            RLRSQActiveDeformationGradientModel(0.75)
        ]
        @testset failfast=true "Active Stress" for active_stress in active_stress_set
            @testset for contraction_model in contraction_model_set
                model = ActiveStressModel(
                    passive_spring,
                    active_stress,
                    CaDrivenInternalSarcomereModel(contraction_model, ConstantCoefficient(1.0)),
                    fsncoeff
                )
                @test_opt Thunderbolt.stress_function(model, F, fsneval, Caᵢ)
                @test Thunderbolt.stress_function(model, F, fsneval, Caᵢ)≈zero(Tensor{2, 3}) atol=1e-16
                @test_opt Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                P, 𝔸 = Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                @test P≈zero(Tensor{2, 3}) atol=1e-16
            end
        end


        @testset failfast=true "Generalized Hill" begin
            @testset "Active Deformation Gradient" for Fᵃmodel in Fᵃmodel_set
                @testset "Contraction Model" for contraction_model in contraction_model_set
                    model = GeneralizedHillModel(
                        passive_spring,
                        ActiveMaterialAdapter(passive_spring),
                        Fᵃmodel,
                        CaDrivenInternalSarcomereModel(contraction_model, ConstantCoefficient(1.0)),
                        fsncoeff
                    )
                    @test_opt Thunderbolt.stress_function(model, F, fsneval, Caᵢ)
                    @test Thunderbolt.stress_function(model, F, fsneval, Caᵢ)≈zero(Tensor{2, 3}) atol=1e-16
                    @test_opt Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                    P, 𝔸 = Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                    @test P≈zero(Tensor{2, 3}) atol=1e-16
                end
            end
        end
        @testset failfast=true "Extended Hill" begin
            @testset "Active Deformation Gradient" for Fᵃmodel in Fᵃmodel_set
                @testset "Contraction Model" for contraction_model in contraction_model_set
                    model = ExtendedHillModel(
                        passive_spring,
                        ActiveMaterialAdapter(passive_spring),
                        Fᵃmodel,
                        CaDrivenInternalSarcomereModel(contraction_model, ConstantCoefficient(1.0)),
                        fsncoeff
                    )
                    @test_opt Thunderbolt.stress_function(model, F, fsneval, Caᵢ)
                    @test Thunderbolt.stress_function(model, F, fsneval, Caᵢ)≈zero(Tensor{2, 3}) atol=1e-16
                    @test_opt Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                    P, 𝔸 = Thunderbolt.stress_and_tangent(model, F, fsneval, Caᵢ)
                    @test P≈zero(Tensor{2, 3}) atol=1e-16
                end
            end
        end
    end

    @testset "Cell Model $model" for model in [Thunderbolt.FHNModel(), Thunderbolt.PCG2019()]
        du = Thunderbolt.default_initial_state(model)
        u = copy(du)
        @test_opt Thunderbolt.cell_rhs!(du, u, nothing, 0.0, model)
    end
end
