---
name: quantica-style
description: Coding style, conventions, testing and workflow for the Quantica.jl codebase. Load before writing, reviewing or testing any code in Quantica, or drafting a Quantica PR or commit message.
---

# Quantica.jl: style and conventions

These are the maintainer's preferences, learned over several PRs. Follow them unless they say
otherwise. When unsure, copy what the surrounding code does.

## Workflow

- **The maintainer commits.** Don't commit, push or open PRs unless asked. Before they commit,
  list anything they should know: behavior changes, stale commit messages, commits worth squashing,
  and anything checked by hand only, such as doctests.
- **Never edit `Manifest.toml` by hand.** Change it only through Pkg commands.
- **One track per conversation, small focused PRs.** If you find something unrelated, note it as a
  follow-up instead of fixing it now.
- **Keep PR messages short.** Say what changed and why, and add a caveat if there is one.
- **Don't rewrite the maintainer's docstrings or comments.** They often edit them after you. When
  reviewing their changes, check them, but don't reword their text.
- **Don't invent references.** No made-up issue or PR numbers in comments.

## Code style

### Field access
- **Use accessors, not `x.field`.** Outside the file that defines a type, call an accessor function
  instead. If one is missing, add it next to the type, for example in the `## API ##` region of
  `types.jl`. This applies to tests too.
- **Bind fields at the top of the function.** Unpack them in one line, then use the locals:
  `(; h, solver, mat, ws, transform) = s` or `Φ, Χ = s.tmp.DD, s.tmp.DD´`.
  Never write nested getfield chains like `s.tmp.ws.x` inside calls.

### Names and syntax
- **Primes are `´`, not `'`.** Write variants as `φs´`, `ws´`, `s´` (`'` is adjoint).
- **Use short-form definitions,** including for local functions:
  `f(o, args::Vararg{Any,M}; kw...) where {M} = ...`. Never `f = function ... end`.
- **Throw errors with `argerror("...")`** (from `tools.jl`), with a message that tells the user what
  went wrong and what was received.

### File organization
- **Organize files with regions.** Each one opens with a header of `#` characters, then the title,
  then `#region`, and closes with `#endregion`:
  ```
  ############################################################################################
  # Title
  #region
  ...
  #endregion
  ```
  Inside `types.jl`, regions such as `#region ## Constructors ##` and `#region ## API ##` hold
  the accessors.
- **Keep lines around 100 characters wide.** Comments should explain *why*, not what.
- **Keep comments inside function bodies short.** Prefer a brief trailing comment, e.g.
  `# complex, for type stability`. Save longer explanations for a comment block above the
  function or region.

## Performance conventions

- **Avoid allocations in hot paths.** The goal is often exactly zero. Check with `@btime` /
  `@allocations`, and compare against a hand-rolled baseline when that's relevant.
- **`call!` / `copy` pattern.** `Quantica.call!(x, args...; params...)` writes into buffers inside
  `x` and returns them aliased. The user-facing `x(args...; params...)` is
  `copy(call!(x, args...; params...))`.
  - Anything with internal buffers needs `minimal_callsafe_copy`, so that `call!` on a copy never
    affects the original. This matters for threading, e.g. one solver per thread in `bands`.
  - `call!` is unexported. Use it as `Quantica.call!`.
- **Workspaces.**
  - LAPACK workspaces come from FastLapackInterface.jl (`HermitianEigenWs`, `EigenWs`, `SVDsddWs`,
    ...).
  - The singleton `MissingWorkspace()` marks a type with no workspace, such as non-BLAS `Float16`,
    and dispatches to an allocating fallback.
  - The `*_ws!` helpers live in the "LAPACK tools and workspaces" region of `tools.jl`.
  - Return LAPACK buffers directly. Don't copy them, and don't reach into a workspace's internal
    fields from outside its file.
- **Type stability.** A variable must have the same concrete type in every branch: no Union
  splitting. For example, copy real eigenvalues into a complex buffer so that all branches return
  `Vector{C}`.
- **Specialization.**
  - Julia does not specialize on a pass-through `args...`. Force it with
    `args::Vararg{Any,N} where {N}` (and `f::F where {F}` for function arguments).
  - Over heterogeneous tuples, use `map`, which Base unrolls. Not `foreach`, which loops at
    runtime and allocates. Not `first`/`tail` or varargs recursion, which hit inference limits on
    Julia 1.10.
  - **Never use `@generated`.**
- **Micro-optimizations that worked:**
  - `abs2` instead of `abs` in tolerance checks (`atol2 = rtol^2 * maximum(abs2, A)`).
  - Column-major loop order.
  - Write `(a < b) & (b < c)` instead of chained comparisons in broadcasts.

## API and docs

- **New API starts unexported,** used as `Quantica.diagonalizer`, `Quantica.call!`, and so on.
  The maintainer decides what becomes public.
- **Docstrings live in `src/docstrings.jl`,** not next to the code.
- **Display methods** go in `src/show.jl`, as a `summary` line plus a `show` with aligned
  `"  Label           : value"` rows that respect `get(io, :indent, "")`. Don't add `show`
  methods for internals that users never see, such as workspaces.
- **Docs pages** live in `docs/src/...` and are registered in `docs/make.jl`. Plain ```` ```julia ````
  blocks are not run, so use them for expensive examples such as benchmarks that should stay out
  of CI.

## Testing

- **CI** runs Julia `1`, `lts` (1.10) and `pre`, using ParallelTestRunner (`test/runtests.jl`).
  - Each `test/test_*.jl` file runs in its own sandbox module, with `test/testutils.jl` loaded.
  - **Code must work on 1.10.** Check it locally on 1.10 as well as on the latest release
    (`julia +1.10`).
- **Allocation tests** use the helper in `testutils.jl`, never a bare `@allocations` in test scope.
  On 1.10, test-scope variables add spurious allocations.
  ```julia
  @test allocs(Quantica.call!, h, φ; params...) == 0
  ```
  `allocs` compiles `f` first, then measures inside a function, and passes keyword arguments
  through. Prefer `== 0` over loose bounds when the true count is zero.
- **Display tests** go in `test/test_show.jl` as `@test nothing === show(stdout, x)`.
- **Coverage.** Add light tests for new `show` methods and error paths:
  `@test_throws ArgumentError ...`.
- **Run test files individually, sandboxed like CI.** Run from `test/`, since some files read data
  files there:
  ```julia
  using Test, Quantica
  for file in ARGS
      m = Module(Symbol(file))
      Core.eval(m, :(using Test, Quantica))
      Base.include(m, "testutils.jl")
      @testset "$file" begin Base.include(m, file * ".jl") end
  end
  ```
  - **On 1.10:** use a scratch environment with `Pkg.develop(path = "<Quantica clone>")` plus
    the test dependencies (e.g. ArnoldiMethod, ParallelTestRunner).
  - **Full suite:** `julia --project -e 'using Pkg; Pkg.test()'`.
- **Known baseline:** a couple of `@test_broken` results are pre-existing. Report pass, broken and
  total counts faithfully.

## Code map

| Area | File |
|---|---|
| Types, constructors, accessors (`## API ##`), `minimal_callsafe_copy` | `src/types.jl` |
| `apply(...)`: turns user-facing objects into applied ones (e.g. `AppliedEigenSolver`) | `src/apply.jl` |
| `call!` for Hamiltonians, `applymodifiers!` | `src/hamiltonian.jl` |
| `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `modifier(term)` | `src/models.jl` |
| `bands`, `spectrum`, `diagonalizer`, `BerryCurvature` | `src/bands.jl` |
| Eigensolvers (`ES`): `LinearAlgebraWorkspace`, `eigen_ws!` | `src/solvers/eigensolvers.jl` |
| Green solvers (`GS`): Schur, SparseLU, KPM, Spectrum, Bands | `src/solvers/green/*.jl` |
| Generic helpers, `argerror`, `ishermitian_approx`, LAPACK `*_ws!` | `src/tools.jl` |
| Input sanitizing: `sanitize_SVector`, `sanitize_eigen`, ... | `src/sanitizers.jl` |
