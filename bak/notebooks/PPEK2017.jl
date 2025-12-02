### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 48685583-8831-4797-bde9-f2032ef42837
let
	using Pkg
	Pkg.activate(Base.current_project());

	using Ferrite, Tensors, Thunderbolt, Optim, UnPack, Plots
end

# ╔═╡ f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
md"""
# A viscoactive constitutive modeling framework with variational updates for the myocardium

## Framework Recap

We start by introducing the multiplicative decomposition of the deformation gradient $\mathbf{F} = \mathbf{F}^{\mathrm{e}} \mathbf{F}^{\mathrm{a}}$, assuming that the active component creates a plasticity-like intermediate configuration. Since $\mathbf{F}$ must be invertible way we can express the elastic portion of the deformation gradient via $\mathbf{F}^{\mathrm{e}} = \mathbf{F} {\mathbf{F}^\mathrm{a}}^{-1}$. $\mathbf{F}^{\mathrm{a}}$ is assumed to be computable from the active contraction model.

For the the nonlinear spring we have two Helmholtz free energy density functions, namely $\Psi^{\mathrm{p}}(\mathbf{F})$ for the "passive" spring and the serial spring attached to the active element $\Psi^{\mathrm{s}}(\mathbf{F}^{\mathbf{e}})$. Note that $\mathbf{F}^{\mathrm{e}}$ is not explicitly given, but we can recover it as discussed above, giving us $\Psi^{\mathrm{s}}(\mathbf{F}{\mathbf{F}^{\mathrm{a}}}^{-1}) \equiv \Psi^{\mathrm{s}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}})$. We summarize these potentials under $\Psi^{\mathrm{spring}}(\mathbf{F},\mathbf{F}^{\mathrm{a}})$.

Further we assume that there exists a viscosity potential $\Psi^{\mathrm{v}}(\mathbf{F}, \dot{\mathbf{F}})$, such that a first Piola-Kirchhoff stress tensor can be derived from it via $\mathbf{P}^{\mathrm{v}} = \partial_{\dot{\mathbf{F}}} \Psi^{\mathrm{v}}(\mathbf{F}, \dot{\mathbf{F}})$.

**QUESTION:** Where does this idea originate from? Do the dimensions match?

With this information we can express the Piola-Kirchhoff stress tensor as

```math
\mathbf{P}
= \partial_{\mathbf{F}} \Psi^{\mathrm{spring}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}})
+ \partial_{\dot{\mathbf{F}}} \Psi^{\mathrm{v}}(\mathbf{F}, \dot{\mathbf{F}}) \, .
```

## Variational Constitutive Model

With the vector of internal variables $\mathbf{Q}$, geometric information $\mathbf{M}$ and the thermodynamic force

```math
\mathbf{Y}
= - \partial_{\mathbf{F}^{\mathrm{a}}} \Psi^{\mathrm{spring}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}}) \cdot \mathbf{M} \mathbf{F}^{\mathrm{a}}
  - \partial_{\mathbf{F}^{\mathrm{a}}} \Psi^{\mathrm{spring}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}})
```

we can construct a potential function

```math
D(\dot{\mathbf{F}}, \dot{\mathbf{Q}})
= \partial_{\mathbf{F}} \Psi^{\mathrm{spring}}(\mathbf{F}, \mathbf{F}^{\mathrm{a}}) \cdot \dot{\mathbf{F}}
  - \mathbf{Y} \cdot \dot{\mathbf{Q}}
  + \Psi^{\mathrm{Q}*}(\dot{\mathbf{Q}})
  + \Psi^{\mathrm{v}}(\dot{\mathbf{F}})
```

**QUESTION:** I am lost. :)
"""

# ╔═╡ 94fc407e-d72e-4b44-9d30-d66c433b954a
md"""
## Material Point
"""

# ╔═╡ e5b588b3-8eaa-42d6-a0da-09c60be6ff09
Fᵃ₀ = Tensor{2, 3}((i,j) -> i==j==1 ? 1.2 : Float64(i==j))

# ╔═╡ 8e4159fb-bfb2-4244-bca5-f2e4237c43f5
F₀ = Tensor{2, 3}((i,j) -> i==j==1 ? 1.2 : Float64(i==j))

# ╔═╡ 012f590f-1e33-46ce-9d05-8d2ea96508f2
num_steps = 100

# ╔═╡ 4598450b-449f-4da0-9873-157ece606975
 ΔF = Tensor{2, 3}((i,j) -> Float64(i==j==1)/num_steps*0.2)

# ╔═╡ 86001544-6e16-4d2c-957c-d7af8e04dd62
Qᶠ₀ = 0.0

# ╔═╡ 8fb1520b-77fe-4279-baf1-448389299dbb
η = 0.0

# ╔═╡ d314acc8-22fc-40f8-bcc6-729b6e208f70
struct NeoHookeanModel
	λ
	μ
end

# ╔═╡ bc3bbb45-c263-4f29-aa6f-647e8dbf3466
function ψ(F, Mᶠ, model::NeoHookeanModel)
	@unpack μ, λ = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	I₃ = det(C)
	J = sqrt(I₃)

    return μ / 2 * (I₁ - 3) - μ * log(J) + λ / 2 * log(J)^2
end

# ╔═╡ f4a00f5d-f042-4804-9be1-d24d5046fd0a
struct FiberSheetNormal
	f
	s
	n
end

# ╔═╡ c1a5b7d0-0987-412c-a456-26f2662e831c
struct ModifiedLinYinModel
	D₁
	D₂
	D₃
	D₄
	D₅
end

# ╔═╡ a44b5b0e-84c7-4245-8d1c-54279593c20a
function ψ(F, Mᶠ, model::ModifiedLinYinModel)
	@unpack D₁, D₂, D₃, D₄, D₅ = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	#I₂ = (I₁^2 - tr(C^2))/2.0
	I₃ = det(C)
	J = sqrt(I₃)
	I₄ = C ⊡ Mᶠ # = C : (f ⊗ f)

	# Exponential portion
	D₁⋅(I₁-3)⋅(I₄-1) + D₂⋅(I₁-3)^2 + D₃⋅(I₄-1)^2 + D₄⋅(I₁-3-2⋅log(J)) + D₅/2.0⋅log(J)^2
end

# ╔═╡ 826694fd-ddba-438b-878e-31f999e7c9c1
struct ModifiedHumphreyModel
	C₁
	C₂
	C₃
	C₄
	C₅
	C₆
end

# ╔═╡ b996d803-c1fa-46b9-b682-195bf3b1c770
function ψ(F, Mᶠ, model::ModifiedHumphreyModel)
	@unpack C₁, C₂, C₃, C₄, C₅, C₆ = model

	C = tdot(F) # = FᵀF

	# Invariants
	I₁ = tr(C)
	#I₂ = (I₁^2 - tr(C^2))/2.0
	I₃ = det(C)
	J = sqrt(I₃)
	I₄ = C ⊡ Mᶠ # = C : (f ⊗ f)
	√I₄ = sqrt(I₄)

	C₁⋅(√I₄-1)^2 + C₂⋅(√I₄-1)^3 + C₃⋅(I₁-3)⋅(√I₄-1) + C₄⋅(I₁-3)^2 + C₅⋅(I₁-3-2⋅log(J)) + C₆/2.0⋅log(J)^2
end

# ╔═╡ de9ff8e6-99cc-46e6-b2ec-970027bf4548
function optimize_Qᶠ_automatic(Fₖ₊₁, Fₖ, Fᵃₖ, Qᶠₖ, fsn, Δt)
	# Parameters from Table 1
	modelˢ = ModifiedLinYinModel(-38.70,40.83,25.12,9.51,171.18)
	modelᵖ = ModifiedHumphreyModel(15.98,55.85,-33.27,30.21,3.59,64.62)

	# modelˢ = NeoHookeanModel(4.0, 0.4)
	# modelᵖ = NeoHookeanModel(4.0, 0.4)

	Mᶠ = fsn.f ⊗ fsn.f
	Mˢ = fsn.s ⊗ fsn.s
	Mⁿ = fsn.n ⊗ fsn.n

	# Eq (4)
	A(F,Fᵃ,Q) = ψ(F⋅inv(Fᵃ), Mᶠ, modelˢ) + ψ(F, Mᶠ, modelᵖ)

	# Solve optimization problem in eq (47)
	function f(Qₖ₊₁)
		Qᶠₖ₊₁ = Qₖ₊₁[1]
		ΔQᶠ = Qᶠₖ₊₁ - Qᶠₖ

		# Fᵃₖ₊₁ from flow rule (45)
		Fᵃₖ₊₁ = (exp(ΔQᶠ)*Mᶠ + Mˢ + Mⁿ)⋅Fᵃₖ

		A(Fₖ₊₁, Fᵃₖ₊₁, Qᶠₖ₊₁) - A(Fₖ, Fᵃₖ, Qᶠₖ)
	end

	Qᶠₖ₊₁ = Optim.minimizer(optimize(f, [Qᶠₖ], Newton()))[1]

	Pᵉ₁₁ = gradient(F->ψ(F, Mᶠ, modelᵖ), Fₖ₊₁)[1,1]

	ΔQᶠ = Qᶠₖ₊₁ - Qᶠₖ
	return [Qᶠₖ₊₁, (exp(ΔQᶠ)*Mᶠ + Mˢ + Mⁿ)⋅Fᵃₖ, Pᵉ₁₁]
end

# ╔═╡ 00ca16c9-4eee-40d6-8d3e-ff3527412181
function simulate_material_point_1()
	fsn = FiberSheetNormal(Tensor{1,3,Float64}([1.0,0.0,0.0]), Tensor{1,3,Float64}([0.0,1.0,0.0]), Tensor{1,3,Float64}([0.0,0.0,1.0]))
	Δt = 0.01

	# Initial condition
	Fₖ = F₀
	Fᵃₖ = Fᵃ₀
	Qᶠₖ = Qᶠ₀

	Fs = Vector{Float64}()
	Ps = Vector{Float64}()
	Qs = Vector{Float64}()

	for iter = 1:num_steps
		# Impose deformation gradient
		Fₖ₊₁ = Fₖ + ΔF

		# Update internal state and active deformation gradient
		Qᶠₖ₊₁, Fᵃₖ₊₁, Pᵉ₁₁ = optimize_Qᶠ_automatic(Fₖ₊₁, Fₖ, Fᵃₖ, Qᶠₖ, fsn, Δt)

		append!(Qs, Qᶠₖ₊₁)
		append!(Fs, Fᵃₖ₊₁[1,1])
		append!(Ps, Pᵉ₁₁)

		# Values for next iteration
		Qᶠₖ = Qᶠₖ₊₁
		Fᵃₖ = Fᵃₖ₊₁
		Fₖ = Fₖ₊₁
	end

	l = @layout [a b]
	p1 = plot(1:num_steps, Fs, label="F11", xaxis=:log)
	p2 = plot(1:num_steps, Ps, label="P11", xaxis=:log)
	plot(p1, p2, layout = l)

	# plot(num_steps:(2*num_steps), Qs[num_steps:(2*num_steps)], label="Q")


end

# ╔═╡ 889559dd-cff1-4a4b-bc5d-13bacfa703dc
simulate_material_point_1()

# ╔═╡ c384e267-82c9-406f-ab5c-d63684c29519
function optimize_Qᶠ_manually(Fₖ₊₁, Qᶠₖ, Fᵃₖ, Mᶠ, Δt, newton_tol, max_newton_iter = 100)
	# Initial guess
	Qᶠₖ₊₁ = Qᶠₖ
	for newton_iter = 1:max_newton_iter
		dWdQ = computedWdQ();
		d2WdQ2 = computed2WdQ2();

		ΔQ = -1. * d2WdQ2.inverse() * dWdQ;
		Qᶠₖ₊₁ += ΔQ;

		if norm(ΔQ) < newton_tol
			break
		end
	end

	return Qᶠₖ₊₁
end

# ╔═╡ Cell order:
# ╠═48685583-8831-4797-bde9-f2032ef42837
# ╟─f6b5bde0-bcdc-41de-b64e-29b9e60b06c0
# ╟─94fc407e-d72e-4b44-9d30-d66c433b954a
# ╠═e5b588b3-8eaa-42d6-a0da-09c60be6ff09
# ╠═8e4159fb-bfb2-4244-bca5-f2e4237c43f5
# ╠═012f590f-1e33-46ce-9d05-8d2ea96508f2
# ╠═4598450b-449f-4da0-9873-157ece606975
# ╟─86001544-6e16-4d2c-957c-d7af8e04dd62
# ╟─8fb1520b-77fe-4279-baf1-448389299dbb
# ╟─d314acc8-22fc-40f8-bcc6-729b6e208f70
# ╟─bc3bbb45-c263-4f29-aa6f-647e8dbf3466
# ╟─f4a00f5d-f042-4804-9be1-d24d5046fd0a
# ╟─c1a5b7d0-0987-412c-a456-26f2662e831c
# ╟─a44b5b0e-84c7-4245-8d1c-54279593c20a
# ╟─826694fd-ddba-438b-878e-31f999e7c9c1
# ╟─b996d803-c1fa-46b9-b682-195bf3b1c770
# ╠═de9ff8e6-99cc-46e6-b2ec-970027bf4548
# ╠═00ca16c9-4eee-40d6-8d3e-ff3527412181
# ╠═889559dd-cff1-4a4b-bc5d-13bacfa703dc
# ╟─c384e267-82c9-406f-ab5c-d63684c29519
