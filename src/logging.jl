# TODO we should follow the pattern in ProgressLogging.jl and provide consumable
#      log messages in this package, which are then processed by some new global logger
Base.@kwdef struct DefaultProgressMonitor
    # id::Symbol
    # msgs::TerminalLoggers.StickyMessages
end

# function DefaultProgressMonitor(id::Symbol)
#     DefaultProgressMonitor(id, TerminalLoggers.StickyMessages())
# end

# function DefaultProgressMonitor(id::Symbol, outer_monitor::DefaultProgressMonitor)
#     DefaultProgressMonitor(id, outer_monitor.msgs)
# end

#

function integration_step_monitor(integrator::SciMLBase.DEIntegrator, progress_monitor::DefaultProgressMonitor)
    # (; id)       = progress_monitor
    (; t, dt, iter)    = integrator
    # push!(msgs, id => "$id: integrating on [$t, $(t+dt)] with Δt=$dt.")
    @logmsg LogLevel(-100) "Integrating on [$t, $(t+dt)]." iter=iter Δt=dt _group=:timeintegration
end

function integration_finalize_monitor(integrator, progress_monitor::DefaultProgressMonitor)
    # (; id)         = progress_monitor
    (; t, iter) = integrator
    # push!(msgs, id => "$id: done at $t.")
    @info "Finished integration at t=$t." iter=iter _group=:timeintegration
end

#

function nonlinear_step_monitor(nlcache, time, f, progress_monitor::DefaultProgressMonitor)
    # (; id, msgs) = progress_monitor
    (; iter, linear_solver_cache) = nlcache
    stats = hasproperty(linear_solver_cache.cacheval, :stats) ? linear_solver_cache.cacheval.stats : nothing
    # push!(msgs, id => "$id: $(nlcache.iter)\n\t||r||=$(norm(nlcache.residual)) ||Δu||=$(norm(linear_solver_cache.u))\n\t$stats")
    resnorm = norm(nlcache.residual)
    normΔu = norm(linear_solver_cache.u)
    if stats === nothing
        @logmsg LogLevel(-100) "Nonlinear solve step" iter=iter resnorm=resnorm normΔu=normΔu _group=:nlsolve
    else
        @logmsg LogLevel(-100) "Nonlinear solve step" iter=iter stats=stats _group=:nlsolve
    end
end

function nonlinear_finalize_monitor(nlcache, time, f, progress_monitor::DefaultProgressMonitor)
    # (; id, msgs) = progress_monitor
    # push!(msgs, id => "$id: done.")
end

#

function linear_finalize_monitor(lincache, progress_monitor::DefaultProgressMonitor, sol)
    stats = hasproperty(lincache.cacheval, :stats) ? lincache.cacheval.stats : nothing
    success = DiffEqBase.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == DiffEqBase.ReturnCode.Default
    if stats === nothing
        @logmsg LogLevel(-100) "Solved system with matrix type $(typeof(lincache.A)) of size $(size(lincache.A))." success=success _group=:linsolve
    else
        @logmsg LogLevel(-100) "Solved system with matrix type $(typeof(lincache.A)) of size $(size(lincache.A))." success=success stats=stats _group=:linsolve
    end
end


Base.@kwdef struct VTKNewtonMonitor{MonitorType}
    outdir::String
    inner_monitor::MonitorType = DefaultProgressMonitor()
end

function nonlinear_step_monitor(cache, time, f, monitor::VTKNewtonMonitor)
    nonlinear_step_monitor(cache,time,f,monitor.inner_monitor)

    VTKGridFile(joinpath(monitor.outdir, "newton-monitor-t=$time-i=$newton_itr.vtu"), f.dh) do vtk
        write_solution(vtk, f.dh, u)
        write_solution(vtk, f.dh, linear_cache.b, "_residual")
        write_solution(vtk, f.dh, linear_cache.u, "_increment")
    end
end

function nonlinear_finalize_monitor(nlcache, time, f, progress_monitor::VTKNewtonMonitor)
end
