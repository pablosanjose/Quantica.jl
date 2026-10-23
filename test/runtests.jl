using ParallelTestRunner
using Quantica

# Each test file runs concurrently in its own sandbox module (see ParallelTestRunner.jl docs)
testsuite = find_tests(@__DIR__)
delete!(testsuite, "testutils")  # shared helpers, not a test file
Sys.WORD_SIZE == 64 || delete!(testsuite, "test_plots")

init_code = quote
    using Quantica
    include($(pkgdir(Quantica, "test", "testutils.jl")))
end

# These files check errors thrown before an extension's trigger package is loaded
# (e.g. qplot without a Makie backend). Workers are reused across files and packages
# can't be unloaded, so these files get a dedicated new worker.
const NEEDS_FRESH_WORKER = ("test_bandstructure", "test_plots")

runtests(Quantica, ARGS; testsuite, init_code,
    test_worker = test -> test in NEEDS_FRESH_WORKER ? addworker() : nothing)
