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

# ParallelTestRunner sizes its worker pool as min(cores, available memory ÷ 2 GiB), but available
# memory is a noisy snapshot on macOS (a 7 GB GitHub runner sometimes got a single worker).
# Use total memory instead, which is fixed. Set PTR_NUM_JOBS or pass --jobs=N to override.
if Sys.isapple() && !haskey(ENV, "PTR_NUM_JOBS")
    ENV["PTR_NUM_JOBS"] = string(clamp(Sys.total_memory() ÷ (2 * 2^30), 1, Sys.CPU_THREADS))
end

# These files check errors thrown before an extension's trigger package is loaded
# (e.g. qplot without a Makie backend). Workers are reused across files and packages
# can't be unloaded, so these files get a dedicated new worker.
const NEEDS_FRESH_WORKER = ("test_bandstructure", "test_plots")

runtests(Quantica, ARGS; testsuite, init_code,
    test_worker = test -> test in NEEDS_FRESH_WORKER ? addworker() : nothing)
