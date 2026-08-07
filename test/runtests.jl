using Thunderbolt
using ParallelTestRunner

# Each test file runs in its own module, in its own worker process. That is why every file under
# `test/` carries its own `using` header and includes `testfixtures.jl` itself — keep it that way, or
# it will pass here and fail when run on its own (and vice versa).
#
# Useful invocations:
#   julia --project=. -e 'using Pkg; Pkg.test()'                         # all files, parallel
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["--jobs=1"])'   # serial, for debugging
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_mesh"])'  # one file (prefix match)
#   julia --project=. -e 'using Pkg; Pkg.test(test_args=["--list"])'     # show what would run

const TESTDIR = @__DIR__

testsuite = find_tests(TESTDIR)   # NOT the `pwd()` default: from the repo root that walks docs/, bak/, …

# Not part of the suite:
#   testfixtures    — shared helpers, included by the files that need them
#   gpu/*           — needs CUDA and its own project, no CI job yet
#   test_fsi_element — scratch file, untracked and without a testset
delete!(testsuite, "testfixtures")
for name in collect(keys(testsuite))
    startswith(name, "gpu/") && delete!(testsuite, name)
end

# `default_njobs()` maximises workers against CPU count and free memory, but it does not know that the
# integration workers below each take `INTEGRATION_THREADS` threads for the per-color assembly. The
# useful bound is therefore cores ÷ threads-per-worker, not cores. Measured on a 16-core box:
# 4 jobs = 190 s, the 11-job default = 340 s, serial = 520 s. Explicit `--jobs=N` still wins.
const INTEGRATION_THREADS = max(1, min(4, Sys.CPU_THREADS ÷ 4))
default_jobs() = clamp(Sys.CPU_THREADS ÷ INTEGRATION_THREADS, 1, 4)

argv = copy(ARGS)
any(startswith("--jobs"), argv) || push!(argv, "--jobs=$(default_jobs())")

args = parse_args(argv)
filter_tests!(testsuite, args)

# Loaded once per test file, into the fresh module that file gets.
const init_code = quote
    using Test
    using Thunderbolt
end

# The integration tests are the ones that exercise the threaded per-color assembly, so give those
# workers real threads. `addworker` otherwise pins JULIA_NUM_THREADS=1, which would silently drop
# that coverage; `-t` overrides the env var. Keep the product of jobs x threads at or below the core
# count — Polyester spins, so oversubscription hurts more than it helps.
test_worker(name) =
    if startswith(name, "integration/")
        addworker(; exeflags = ["--threads=$(INTEGRATION_THREADS)"])
    else
        nothing
    end

runtests(Thunderbolt, args; testsuite, init_code, test_worker)
