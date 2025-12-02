using BenchmarkTools, Thunderbolt, StaticArrays
cell_model = Thunderbolt.PCG2019()
solver_cache = Thunderbolt.ForwardEulerCellSolverCache(zeros(7), zeros(1), zeros(1,6))
@btime Thunderbolt.perform_step!($cell_model, 0.0, 0.1, $solver_cache)
@code_warntype Thunderbolt.perform_step!(cell_model, 0.0, 0.1, solver_cache)


@btime Thunderbolt.cell_rhs!($solver_cache.du, $solver_cache.uₙ[1], $solver_cache.sₙ[1,:], nothing, 0.0, $cell_model)
@code_warntype Thunderbolt.cell_rhs_fast!(solver_cache.du, solver_cache.uₙ[1], solver_cache.sₙ[1,:], nothing, 0.0, cell_model)

cell_model = Thunderbolt.FHNModel()
solver_cache = Thunderbolt.ForwardEulerCellSolverCache(zeros(2), zeros(1), zeros(1,1))
@btime Thunderbolt.perform_step!($cell_model, 0.0, 0.1, $solver_cache)
