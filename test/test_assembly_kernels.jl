using Test, Thunderbolt, Tensors
import Thunderbolt: FerriteOperators

# The kernels below exploit structure: a vectorized scalar interpolation's gradients have one nonzero
# row, and a shape function which does not live on a facet vanishes there. Each is held against the
# plain double loop over all shape functions.

# A distorted element, so that no entry vanishes by symmetry of the geometry.
distorted(x::Vec{3}) = Vec((x[1] + 0.1x[2] * x[3], x[2] - 0.05x[1]^2, 1.2x[3] + 0.1x[1]))
distorted_coordinates(cell_type) = [
    distorted(Vec{3}(ξ)) for
    ξ in Ferrite.reference_coordinates(Ferrite.geometric_interpolation(cell_type))
]

@testset "Assembly kernels" begin
    @testset "Quadrature point kernel on $(ip) in 3D" for (ip, cell_type) in (
        (Lagrange{RefHexahedron, 1}(), Hexahedron),
        (Lagrange{RefHexahedron, 2}(), Hexahedron),
        (Lagrange{RefTetrahedron, 2}(), Tetrahedron),
        (Lagrange{RefPrism, 1}(), Wedge),
    )
        refshape = Ferrite.getrefshape(ip)
        cv = CellValues(
            QuadratureRule{refshape}(3),
            ip^3,
            Ferrite.geometric_interpolation(cell_type),
        )
        Ferrite.reinit!(cv, distorted_coordinates(cell_type))
        n = getnbasefunctions(cv)
        P = rand(Tensor{2, 3})
        tangent = rand(Tensor{4, 3})
        dΩ = 0.37

        K, r = zeros(n, n), zeros(n)
        K₀, r₀ = zeros(n, n), zeros(n)
        for qp in QuadratureIterator(cv)
            Thunderbolt._qs_assemble_qp!(
                FerriteOperators.JacobianResidualRequest(K, r),
                cv,
                qp,
                P,
                tangent,
                dΩ,
            )
            # `nothing` is not a vectorized interpolation, so this is the generic method.
            Thunderbolt._qs_assemble_qp!(
                FerriteOperators.JacobianResidualRequest(K₀, r₀),
                nothing,
                cv,
                qp,
                P,
                tangent,
                dΩ,
            )
        end
        @test K ≈ K₀ rtol = 1e-14
        @test r ≈ r₀ rtol = 1e-14

        # A request without a residual accumulates into the matrix alone.
        K_only = zeros(n, n)
        req = FerriteOperators.JacobianRequest{:u}(K_only)
        for qp in QuadratureIterator(cv)
            Thunderbolt._qs_assemble_qp!(req, cv, qp, P, tangent, dΩ)
        end
        @test K_only ≈ K₀ rtol = 1e-14

        qp = first(QuadratureIterator(cv))
        @test (@allocated Thunderbolt._qs_assemble_qp!(req, cv, qp, P, tangent, dΩ)) == 0
    end

    @testset "Quadrature point kernel falls back in 2D" begin
        cv = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}()^2)
        Ferrite.reinit!(cv, [Vec((0.0, 0.0)), Vec((1.0, 0.1)), Vec((1.1, 1.0)), Vec((0.0, 1.2))])
        n = getnbasefunctions(cv)
        P, tangent = rand(Tensor{2, 2}), rand(Tensor{4, 2})
        K, r = zeros(n, n), zeros(n)
        qp = first(QuadratureIterator(cv))
        Thunderbolt._qs_assemble_qp!(
            FerriteOperators.JacobianResidualRequest(K, r),
            cv,
            qp,
            P,
            tangent,
            1.0,
        )
        for i = 1:n
            ∇δui = shape_gradient(cv, qp, i)
            @test r[i] ≈ ∇δui ⊡ P
            @test all(K[i, j] ≈ ∇δui ⊡ tangent ⊡ shape_gradient(cv, qp, j) for j = 1:n)
        end
    end

    @testset "Facet kernels on order $order" for order in (1, 2)
        ip = Lagrange{RefHexahedron, order}()
        fv = FacetValues(
            FacetQuadratureRule{RefHexahedron}(2order),
            ip^3,
            Lagrange{RefHexahedron, 1}(),
        )
        grid = generate_grid(Hexahedron, (1, 1, 1))
        Ferrite.transform_coordinates!(grid, distorted)
        cell = first(CellIterator(grid))
        n = getnbasefunctions(fv)
        uₑ = 0.05 .* (rand(n) .- 0.5)
        dof_range = 1:n
        lfi = 2
        Ferrite.reinit!(fv, cell, lfi)

        function textbook_springs(∂Ψ∂u, ∂²Ψ∂u²)
            K, r = zeros(n, n), zeros(n)
            for qp in QuadratureIterator(fv)
                dΓ = getdetJdV(fv, qp)
                N = getnormal(fv, qp)
                u_q = function_value(fv, qp, uₑ)
                for i = 1:n
                    δuᵢ = shape_value(fv, qp, i)
                    r[i] += δuᵢ ⋅ ∂Ψ∂u(u_q, N) * dΓ
                    for j = 1:n
                        K[i, j] += (δuᵢ ⋅ ∂²Ψ∂u²(u_q, N) ⋅ shape_value(fv, qp, j)) * dΓ
                    end
                end
            end
            return K, r
        end

        args = (; cell, states = (; u = uₑ), ctx = nothing)
        @testset "$(nameof(typeof(bc)))" for (bc, ∂Ψ∂u, ∂²Ψ∂u²) in (
            (RobinBC(1.7, "x"), (u, N) -> 2 * 1.7 * u, (u, N) -> 2 * 1.7 * one(Tensor{2, 3})),
            (NormalSpringBC(1.7, "x"), (u, N) -> 1.7 * (u ⋅ N) * N, (u, N) -> 1.7 * N ⊗ N),
        )
            cache = Thunderbolt.SimpleFacetCache(bc, fv, dof_range)
            K₀, r₀ = textbook_springs(∂Ψ∂u, ∂²Ψ∂u²)

            K, r = zeros(n, n), zeros(n)
            FerriteOperators.assemble_facet!(
                FerriteOperators.JacobianResidualRequest(K, r),
                cache,
                args,
                lfi,
            )
            @test K ≈ K₀ rtol = 1e-13
            @test r ≈ r₀ rtol = 1e-13

            K_weighted = zeros(n, n)
            req = FerriteOperators.WeightedJacobianRequest(K_weighted, (; u = 0.5))
            FerriteOperators.assemble_facet!(req, cache, args, lfi)
            @test K_weighted ≈ 0.5 * K₀ rtol = 1e-13

            r_only = zeros(n)
            FerriteOperators.assemble_facet!(
                FerriteOperators.ResidualRequest(r_only),
                cache,
                args,
                lfi,
            )
            @test r_only ≈ r₀ rtol = 1e-13
        end

        @testset "follower pressure" begin
            p = 1.3
            K₀, r₀ = zeros(n, n), zeros(n)
            for qp in QuadratureIterator(fv)
                dΓ = getdetJdV(fv, qp)
                n₀ = getnormal(fv, qp)
                ∇u = function_gradient(fv, qp, uₑ)
                F = one(∇u) + ∇u
                invF = inv(F)
                cofF = transpose(invF)
                J = det(F)
                for i = 1:n
                    δuᵢ = shape_value(fv, qp, i)
                    r₀[i] += p * J * cofF ⋅ n₀ ⋅ δuᵢ * dΓ
                    for j = 1:n
                        ∇δuⱼ = shape_gradient(fv, qp, j)
                        δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
                        δJ = J * tr(∇δuⱼ ⋅ invF)
                        K₀[i, j] += p * ((δJ * cofF + J * δcofF) ⋅ n₀) ⋅ δuᵢ * dΓ
                    end
                end
            end

            K, r = zeros(n, n), zeros(n)
            K_weighted = zeros(n, n)
            r_only = zeros(n)
            for qp in QuadratureIterator(fv)
                Thunderbolt.assemble_facet_pressure_qp!(
                    FerriteOperators.JacobianResidualRequest(K, r),
                    uₑ,
                    p,
                    qp,
                    fv,
                    dof_range,
                )
                Thunderbolt.assemble_facet_pressure_jacobian_qp!(
                    (; K = K_weighted),
                    uₑ,
                    p,
                    qp,
                    fv,
                    dof_range,
                    0.5,
                )
                Thunderbolt.assemble_facet_pressure_qp!(
                    FerriteOperators.ResidualRequest(r_only),
                    uₑ,
                    p,
                    qp,
                    fv,
                    dof_range,
                )
            end
            @test K ≈ K₀ rtol = 1e-13
            @test r ≈ r₀ rtol = 1e-13
            @test K_weighted ≈ 0.5 * K₀ rtol = 1e-13
            @test r_only ≈ r₀ rtol = 1e-13
        end

        @testset "the dashpots" begin
            η = 2.3
            cache = Thunderbolt.ViscousFacetCache(ViscousRobinBC(η, "x"), fv, dof_range)
            vₑ = 0.1 .* (rand(n) .- 0.5)
            vargs = (; cell, states = (; v = vₑ), ctx = nothing)
            K₀, r₀ = zeros(n, n), zeros(n)
            for qp in QuadratureIterator(fv)
                dΓ = getdetJdV(fv, qp)
                v_q = function_value(fv, qp, vₑ)
                for i = 1:n
                    δuᵢ = shape_value(fv, qp, i)
                    r₀[i] += η * (δuᵢ ⋅ v_q) * dΓ
                    for j = 1:n
                        K₀[i, j] += η * (δuᵢ ⋅ shape_value(fv, qp, j)) * dΓ
                    end
                end
            end
            K, r = zeros(n, n), zeros(n)
            FerriteOperators.assemble_facet!(FerriteOperators.ResidualRequest(r), cache, vargs, lfi)
            FerriteOperators.assemble_facet!(
                FerriteOperators.WeightedJacobianRequest(K, (; v = 1.0)),
                cache,
                vargs,
                lfi,
            )
            @test K ≈ K₀ rtol = 1e-13
            @test r ≈ r₀ rtol = 1e-13
        end
    end
end
