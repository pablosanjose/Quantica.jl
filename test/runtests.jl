using ParallelTestRunner
using Quantica

# Each test file runs concurrently in its own sandbox module (see ParallelTestRunner.jl docs)
# For optimal test times, make each test file's runtime more or less the same
testsuite = find_tests(@__DIR__)
delete!(testsuite, "testutils")  # shared helpers, not a test file
Sys.WORD_SIZE == 64 || delete!(testsuite, "test_plots")

# Optional test group, set by CI so that only one job loads (and compiles) the Makie stack:
# "plots" runs only test_plots, "main" runs everything else, "all" (default) runs everything
const TEST_GROUP = get(ENV, "QUANTICA_TEST_GROUP", "all")
TEST_GROUP in ("all", "main", "plots") || error("Unknown QUANTICA_TEST_GROUP = $TEST_GROUP")
TEST_GROUP == "plots" && filter!(t -> first(t) == "test_plots", testsuite)
TEST_GROUP == "main" && delete!(testsuite, "test_plots")

init_code = quote
    using Quantica
    include($(pkgdir(Quantica, "test", "testutils.jl")))
    # Discard all output of tests (progress bars, show, warnings) to keep CI logs clean.
    # Test failures are still reported in full by the runner.
    redirect_stdout(devnull); redirect_stderr(devnull)
end

# These files check errors thrown before an extension's trigger package is loaded
# (e.g. qplot without a Makie backend). Workers are reused across files and packages
# can't be unloaded, so these files get a dedicated new worker.
const NEEDS_FRESH_WORKER = ("test_bandstructure", "test_plots")

# TEMPORARY: diagnose PTR's default worker count on macOS CI runners
if Sys.isapple() && haskey(ENV, "GITHUB_ACTIONS")
    @info "PTR worker inputs" Sys.CPU_THREADS get(ENV, "JULIA_CPU_THREADS", nothing) (@static isdefined(Sys, :EFFECTIVE_CPU_THREADS) ? Sys.EFFECTIVE_CPU_THREADS : nothing) ParallelTestRunner.available_memory()/2^30 Sys.free_memory()/2^30 Sys.total_memory()/2^30 ParallelTestRunner.default_njobs()
    run(`vm_stat`)
    run(pipeline(`ps -axm -o rss=,comm=`, `head -12`))
end

runtests(Quantica, ARGS; testsuite, init_code,
    test_worker = test -> test in NEEDS_FRESH_WORKER ? addworker() : nothing)
