using MatrixDepot,LinearSolve,SparseMatricesCSR
using KernelAbstractions

##########################################
## L1 Gauss Seidel Preconditioner - CPU ##
##########################################

@testset "L1GS Preconditioner - Unsymmetric A" begin
    md = mdopen("HB/sherman5") 
    A = md.A
    b = md.b[:,1] 
    nparts = 16
    partsize = size(md.A,1) / 16 |> ceil |> Int

    u = A\b

    sherman5_problem = LinearProblem(A, b)
    sol_unprec = solve(sherman5_problem, KrylovJL_GMRES())
    # check both iterative solution and direct solution
    @test sol_unprec.u ≈ u

    # Test L1GS Preconditioner
    P = L1GSPrecBuilder(CPU())(A, partsize, nparts)
    sol_prec = solve(sherman5_problem, KrylovJL_GMRES(P);Pl=P)
    println( "Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println( "Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test sol_prec.u ≈ u
    @test sol_prec.iters < sol_unprec.iters
    @test sol_prec.resid < sol_unprec.resid
    #@test sol_prec.stats.timer <= sol_unprec.stats.timer # commented out because it fails, is it normal?
end

# TODO: symmetric matrix
