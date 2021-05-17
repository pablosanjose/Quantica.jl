#######################################################################
# Green function
#######################################################################
abstract type AbstractGreensSolver end

struct GreensFunction{S<:AbstractGreensSolver,L,B<:NTuple{L,Union{Int,Missing}},H<:Hamiltonian}
    solver::S
    h::H
    boundaries::B
end

"""
    greens(h::Hamiltonian, solveobject; boundaries::NTuple{L,Integer} = missing)

Construct the Green's function `g::GreensFunction` of `L`-dimensional Hamiltonian `h` using
the provided `solveobject`. Currently valid `solveobject`s are

- the `Bandstructure` of `h` (for an unbounded `h` or an `Hamiltonian{<:Superlattice}}`)
- the `Spectrum` of `h` (for a bounded `h`)
- `Schur1D(; direct = false)` (single-shot generalized [or direct if `direct = true`] eigenvalue approach for 1D Hamiltonians)

If a `boundaries = (n₁, n₂, ...)` is provided, a reflecting boundary is assumed for each
non-missing `nᵢ` perpendicular to Bravais vector `i` at a cell distance `nᵢ` from the
origin.

    h |> greens(h -> solveobject(h), args...)

Curried form equivalent to the above, giving `greens(h, solveobject(h), args...)`.

    g(ω, cells::Pair)

From a constructed `g::GreensFunction`, obtain the retarded Green's function matrix at
frequency `ω` between unit cells `src` and `dst` by calling `g(ω, src => dst)`, where `src,
dst` are `::NTuple{L,Int}` or `SVector{L,Int}`. If not provided, `cells` default to
`(1, 1, ...) => (1, 1, ...)`.

    g(ω, missing)

If allowed by the used `solveobject`, build an efficient function `cells -> g(ω, cells)`
that can produce the Greens function between different cells at fixed `ω` without repeating
cell-independent parts of the computation.

# Examples

```jldoctest
julia> g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> greens(bandstructure(resolution = 17))
GreensFunction{Bandstructure}: Green's function from a 2D bandstructure
  Matrix size    : 1 × 1
  Element type   : scalar (Complex{Float64})
  Band simplices : 512

julia> g(0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im

julia> m = similarmatrix(g); g(m, 0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im
```

# See also
    `greens!`, `Schur1D`
"""
greens(h::Hamiltonian{<:Any,L}, solverobject; boundaries = filltuple(missing, Val(L))) where {L} =
    GreensFunction(greensolver(solverobject, h), h, boundaries)
greens(solver::Function, args...; kw...) = h -> greens(h, solver(h), args...; kw...)

# solver fallback
greensolver(s::AbstractGreensSolver) = s

# missing cells
(g::GreensFunction)(ω; kw...) = g(ω, default_cells(g); kw...)

# call API fallback
(g::GreensFunction)(ω, cells; kw...) = greens!(similarmatrix(g), g, ω, cells; kw...)

similarmatrix(g::GreensFunction, type = Matrix{blocktype(g.h)}) = similarmatrix(g.h, type)

greens!(matrix, g, ω, cells; kw...) = greens!(matrix, g, ω, sanitize_cells(cells, g); kw...)

default_cells(g::GreensFunction) = _plusone.(g.boundaries) => _plusone.(g.boundaries)

_plusone(::Missing) = 1
_plusone(n) = n + 1

sanitize_cells((cell0, cell1)::Pair{<:Integer,<:Integer}, ::GreensFunction{S,1}) where {S} =
    SA[cell0] => SA[cell1]
sanitize_cells((cell0, cell1)::Pair{<:NTuple{L,Integer},<:NTuple{L,Integer}}, ::GreensFunction{S,L}) where {S,L} =
    SVector(cell0) => SVector(cell1)
sanitize_cells(cells, g::GreensFunction{S,L}) where {S,L} =
    throw(ArgumentError("Cells should be of the form `cᵢ => cⱼ`, with each `c` an `NTuple{$L,Integer}`, got $cells"))

const SVectorPair{L} = Pair{SVector{L,Int},SVector{L,Int}}

Base.size(g::GreensFunction, args...) = size(g.h, args...)
Base.eltype(g::GreensFunction) = eltype(g.h)

flatsize(g::GreensFunction, args...) = flatsize(g.h, args...)
blockeltype(g::GreensFunction) = blockeltype(g.h)
blocktype(g::GreensFunction) = blocktype(g.h)
orbitaltype(g::GreensFunction) = orbitaltype(g.h)
orbitalstructure(g::GreensFunction) = orbitalstructure(g.h)

#######################################################################
# Schur1DGreensSolver
#######################################################################

"""
    Schur1D(; deflation = default_tol(T))

Return a Greens function solver using the generalized eigenvalue approach, whereby given the
energy `ω`, the eigenmodes of the infinite 1D Hamiltonian, and the corresponding infinite
and semi-infinite Greens function can be computed by solving the generalized eigenvalue
equation

    A⋅φχ = λ B⋅φχ
    A = [0 I; -h₊ ω-h₀]
    B = [I 0; 0 h₋]

This is the matrix form of the problem `λ(ω-h₀)φ - h₊φ - λ²h₋φ = 0`, where `φχ = [φ; λφ]`,
and `φ` are `ω`-energy eigenmodes, with (possibly complex) momentum `q`, and eigenvalues are
`λ = exp(-iqa₀)`. The algorithm assumes the Hamiltonian has only `dn = (0,)` and `dn = (±1,
)` Bloch harmonics (`h₀`, `h₊` and `h₋`), so its unit cell will be enlarged before applying
the solver if needed. Bound states in the spectrum will yield delta functions in the density
of states that can be resolved by adding a broadening in the form of a small positive
imaginary part to `ω`.

For performace, the eigenvalue equation may be `deflated', i.e. singular solutions `λ=0,∞`
will be removed within the absolute tolerance specified by the keyword `deflation`. If no
deflation is desired, use `deflation = nothing`.

# Examples
```jldoctest
julia> using LinearAlgebra

julia> h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), (10,10)) |> Quantica.wrap(2);

julia> g = greens(h, Schur1D(), boundaries = (0,))
GreensFunction{Schur1DGreensSolver}: Green's function using the Schur1D method
  Matrix size    : 40 × 40
  Deflated size  : 20 × 20
  Element type   : scalar (ComplexF64)
  Boundaries     : (0,)

julia> tr(g(0.3))
-32.193416068730684 - 3.4399800712973074im
```

# See also
    `greens`
"""
struct Schur1D{R}
    atol::R  # could be missing for default_tol(T)
end

Schur1D(; deflation = missing) = Schur1D(deflation)

struct DeflatorWorkspace{T}
    nn::Matrix{T}
    nr1::Matrix{T}
    nr2::Matrix{T}
    rr0::Matrix{T}
    rr1::Matrix{T}
    rr2::Matrix{T}
    rr3::Matrix{T}
    rr4::Matrix{T}
    mb::Matrix{T}
end

DeflatorWorkspace{T}(n, r) where {T} =
    DeflatorWorkspace(Matrix{T}.(undef,
        ((n, n), (n, r), (n, r), (r, r), (r, r), (r, r), (r, r), (r, r), (n+r, n-r)))...)

struct Deflator{T,M<:AbstractMatrix{T},R<:Real,S}
    hp::M
    hm::M
    hmQ0::M             # h₋*Q0 where Q0 = [rowspace(A0) nullspace(A0)]. h₊ = [R' 0] Q0'. h₋ = Q0 [R; 0]
    R::Matrix{T}        # A0 = [-hRR 0; -hBR  0] * [R'; B' ]. R = orthogonal complement of nullspace(A0) === rowspace(A0)
    Adef::Matrix{T}     # deflated A
    Bdef::Matrix{T}     # deflated B
    Ablock::Matrix{T}   # Adef = Ablock * QR; Ablock = [0 I 0; -hRR gRR⁻¹ gRB⁻¹]
    Bblock::Matrix{T}   # Bdef = Bblock * QR; Bblock = [I 0 0; 0 hRR' hBR']
    Vblock´::Matrix{T}  # # Vblock = [-hBR gBR⁻¹ gBB⁻¹] = [R 0] [QB'; QR']. Vblock´ = Matrix(Vblock')
    ig0::Matrix{T}      # Matrix(-h₀)
    ωshifter::S         # metadata to aid in ω-shifting the relevant A subblocks
    atol::R             # A0, A2 deflation tolerance
    tmp::DeflatorWorkspace{T}
end

struct Schur1DGreensSolver{D<:Union{Deflator,Missing},M} <: AbstractGreensSolver
    h0::M
    hp::M
    hm::M
    unitcells::Int
    deflatorR::D
    deflatorL::D
end

function Base.show(io::IO, g::GreensFunction{<:Schur1DGreensSolver})
    print(io, summary(g), "\n",
"  Matrix size    : $(size(g.solver.h0, 1)) × $(size(g.solver.h0, 2))
  Deflated size  : $(deflated_size_text(g))
  Element type   : $(displayelements(g.h))
  Boundaries     : $(g.boundaries)")
end

function deflated_size_text(g::GreensFunction)
    text = hasdeflator(g.solver) <= 0 ? "No deflation" :
        "$(deflated_size_text(g.solver.deflatorR))"
    return text
end

deflated_size_text(d::Deflator) = "$(size(d.Adef, 1) ÷ 2) × $(size(d.Adef, 2) ÷ 2)"

Base.summary(g::GreensFunction{<:Schur1DGreensSolver}) =
    "GreensFunction{Schur1DGreensSolver}: Green's function using the Schur1D method"

Base.size(s::Schur1DGreensSolver, args...) = size(s.deflatorR, args...)
Base.size(d::Deflator, args...) = size(d.Adef, args...)

hasdeflator(::Schur1DGreensSolver{<:Deflator}) = true
hasdeflator(::Schur1DGreensSolver{Missing}) = false

function greensolver(s::Schur1D, h)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a Schur1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    unitcells = max(1, maximum(har -> abs(first(har.dn)), h.harmonics))
    H = maybeflatten(unitcells == 1 ? h : unitcell(h, (unitcells,)))
    h₊, h₀, h₋ = H[(1,)], H[(0,)], H[(-1,)]
    n = size(H, 1)
    T = complex(blockeltype(H))
    atol = s.atol === missing ? default_tol(T) : s.atol
    deflatorR = Deflator(atol, h₊, h₀, h₋)
    deflatorL = Deflator(atol, h₋, h₀, h₊)
    return Schur1DGreensSolver(h₀, h₊, h₋, unitcells, deflatorR, deflatorL)
end

Deflator(atol::Nothing, As...) = missing

function Deflator(atol::Real, h₊::M, h₀::M, h₋::M) where {M}
    h₊ ≈ h₋' || throw(ArgumentError("Deflation requires mutually adjoint intercell h₊ = h₋'. If you intended to build a non-Hermitian Hamiltonian, please use the undeflated method `Schur(deflation = nothing)`."))
    rowspaceR, _, nullspaceR = fullrank_decomposition_qr(h₊, atol)
    B       = Matrix(nullspaceR)                      # nullspace(A0)
    R       = Matrix(rowspaceR)                       # orthogonal complement of nullspace(h₊)
    h₋Q0    = h₋ * parent(rowspaceR)                  # h₋ * [R B] = h₋ * Q0, needed for Jordan chain
    n       = size(h₀, 2)
    r       = size(R, 2)
    b       = size(B, 2)
    T       = eltype(h₀)
    h₊R     = h₊*R
    hRR     = R'*h₊R
    hBR     = B'*h₊R
    gRR⁻¹   = - R'*h₀*R
    gBB⁻¹   = - B'*h₀*B
    gRB⁻¹   = - R'*h₀*B
    gBR⁻¹   = gRB⁻¹'
    g0⁻¹    = Matrix(-h₀)
    Adef    = Matrix{T}(undef, 2r, 2r)       # Needs to be dense for schur!(Adef, Bdef)
    Bdef    = Matrix{T}(undef, 2r, 2r)       # Needs to be dense for schur!(Adef, Bdef)
    Ablock  = Matrix([0I I spzeros(r, b); -hRR gRR⁻¹ gRB⁻¹])
    Bblock  = Matrix([I 0I spzeros(r, b); 0I hRR' hBR'])
    Vblock´ = Matrix([-hBR gBR⁻¹ gBB⁻¹]')
    ωshifter = diag(gRR⁻¹), (r+1:2r, r+1:2r), diag(gBB⁻¹), (2r+1:2r+b, 1:b), diag(g0⁻¹)
    tmp     = DeflatorWorkspace{T}(n, r)
    return Deflator(h₊, h₋, h₋Q0, R, Adef, Bdef, Ablock, Bblock, Vblock´, g0⁻¹, ωshifter, atol, tmp)
end

## Tools

function shiftω!(d::Deflator, ω)
    diagRR, rowcolA, diagBB, rowcolV, diagg0⁻¹ = d.ωshifter
    for (v, row, col) in zip(diagRR, rowcolA...)
        d.Ablock[row, col] = ω + v
    end
    for (v, row, col) in zip(diagBB, rowcolV...)
        d.Vblock´[row, col] = conj(ω) + v
    end
    for (n, v) in enumerate(diagg0⁻¹)
        d.ig0[n, n] = ω + v
    end
    return d
end

function shiftω!(mat::AbstractMatrix, ω)
    @inbounds for i in axes(mat, 1)
        mat[i, i] += ω
    end
    return mat
end

function orthobasis_decomposition_qr(mat, atol)
    q = pqr(mat)
    basis = getQ(q)
    RP´ = getRP´(q)
    n = size(basis, 2)
    r = nonzero_rows(RP´, atol)
    orthobasis = view(basis, :, 1:r)
    complement = view(basis, :, r+1:n)
    r = view(RP´, 1:r, :)
    return orthobasis, r, complement
end

function fullrank_decomposition_qr(mat, atol)
    rowspace, r, nullspace = orthobasis_decomposition_qr(mat', atol)
    return rowspace, r', nullspace
end

nullspace_qr(mat, atol) = last(fullrank_decomposition_qr(mat, atol))

rowspace_qr(mat, atol) = first(fullrank_decomposition_qr(mat, atol))

## Deflate

function deflate(d::Deflator{T,<:SparseMatrixCSC}) where {T}
    r = size(d.R, 2)
    m = size(d.Vblock´, 1)  # m = 2r+b
    # Vblock´ = [RP´ 0] * Q' = b × 2r+b; Q = [rowspaceV nullspaceV] = 2r+b × 2r+b = m × m
    # nullspaceV is 2r+b × 2r
    Vblock´ = copy!(d.tmp.mb, d.Vblock´)  # ::Matrix{T}
    nullspaceV = getQ(qr!(Vblock´, Val(true)), m-2r+1:m)
    Adef = mul!(d.Adef, d.Ablock, nullspaceV)
    Bdef = mul!(d.Bdef, d.Bblock, nullspaceV)
    Q1 = view(nullspaceV, 1:r, :)
    Q2 = view(nullspaceV, r+1:m, :)
    return Adef, Bdef, Q1, Q2
end

## Solver execution: compute self-energy, with or without deflation
(s::Schur1DGreensSolver)(ω, which = Val(:R)) = schursolver(s, ensurecomplex(ω), which)

ensurecomplex(ω::T) where {T<:Real} = ω + im * default_tol(T)
ensurecomplex(ω::Complex) = ω

function schursolver(s::Schur1DGreensSolver{Missing}, ω, which)
    A = Matrix([ω*I - s.h0 -s.hp; -I 0I])
    B = Matrix([s.hm 0I; 0I -I])
    sch = schur!(A, B)
    Σ = nondeflated_selfenergy(which, s, sch)
    # @show sum(abs.(Σ - s.hm * ((ω*I - s.h0 - Σ) \ Matrix(s.hp))))
    return Σ
end

function nondeflated_selfenergy(::Val{:R}, s, sch)
    n = size(s.h0, 1)
    rmodes = retarded_modes(sch, 0)
    ordschur!(sch, rmodes)
    ϕΛR⁻¹ = view(sch.Z, 1:n, 1:n)
    ϕR⁻¹ = view(sch.Z, n+1:2n, 1:n)
    ΣR = s.hm * ϕΛR⁻¹ / ϕR⁻¹
    return ΣR
end

function nondeflated_selfenergy(::Val{:L}, s, sch)
    n = size(s.h0, 1)
    amodes = advanced_modes(sch, 0)
    ordschur!(sch, amodes)
    ϕΛ⁻¹R⁻¹ = view(sch.Z, 1:n, 1:n)
    ϕR⁻¹ = view(sch.Z, n+1:2n, 1:n)
    ΣL = s.hp * ϕR⁻¹ / ϕΛ⁻¹R⁻¹
    return ΣL
end

nondeflated_selfenergy(::Val{:RL}, s, sch) =
    nondeflated_selfenergy(Val(:R), s, sch), nondeflated_selfenergy(Val(:L), s, sch)

schursolver(s::Schur1DGreensSolver{<:Deflator}, ω, ::Val{:R}) =
    deflated_selfenergy(s.deflatorR, s, ω)

schursolver(s::Schur1DGreensSolver{<:Deflator}, ω, ::Val{:L}) =
    deflated_selfenergy(s.deflatorL, s, ω)

schursolver(s::Schur1DGreensSolver{<:Deflator}, ω, ::Val{:RL}) =
    deflated_selfenergy(s.deflatorR, s, ω), deflated_selfenergy(s.deflatorL, s, ω)

function deflated_selfenergy(d::Deflator{T,M}, s::Schur1DGreensSolver, ω) where {T,M}
    # if r = 0 (absolute deflation), Σ=0
    size(d) == (0, 0) && return fill!(d.tmp.nn, zero(T))
    shiftω!(d, ω)
    A, B, Q1, Q2 = deflate(d)
    # find right-moving eigenvectors with atol < |λ| < 1
    sch = schur!(A, B)
    rmodes = retarded_modes(sch, d.atol)
    nr = sum(rmodes)
    ordschur!(sch, rmodes)
    Zret = view(sch.Z, :, 1:nr)
    R, h₋Q0 = d.R, d.hmQ0
    ## Qs    = [Q1; Q2]; [φR; χR; χB] = Qs * Zret * R11
    ## R'φ   = φR = R'Z11 * R11, where R'Z11 = Q1 * Zret
    ## Q0'*χ = Q0'*φ*Λ = [χR; χB]
    ## h₋χ   = h₋ * Q0 * [χR; χB] = h₋ * Q0 * Q2 * Zret * R11 = Z21 * R11, where Z21 = h₋ * Q0 * Q2 * Zret
    ## R´Z11 = Q1 * Zret
    ## Z21   = h₋Q0 * Q2 * Zret
    R´Z11 = Q1 * Zret
    Z21 = h₋Q0 * Q2 * Zret

    ## add generalized eigenvectors until we span the full R space
    R´source, target = add_jordan_chain(d, R´Z11, Z21)

    ΣR = mul!(d.tmp.nn, rdiv!(target, lu!(R´source)), R')
    # @show sum(abs.(ΣR - s.hm * (((ω * I - s.h0) - ΣR) \ Matrix(s.hp))))

    return ΣR
end

# need this barrier for type-stability (sch.α and sch.β are finicky)
function retarded_modes(sch, atol)
    rmodes = Vector{Bool}(undef, length(sch.α))
    rmodes .= atol .< abs.(sch.α ./ sch.β) .< 1
    return rmodes
end

function advanced_modes(sch, atol)
    rmodes = Vector{Bool}(undef, length(sch.β))
    rmodes .= atol .< abs.(sch.β ./ sch.α) .< 1
    return rmodes
end

function add_jordan_chain(d::Deflator, R´Z11, Z21)
    local R´φg_candidates, source_rowspace
    g, hg, gh, hgh = integrate_out_bulk!(d)
    Σ11 = copy(hgh)
    Σn1 = copy(Σ11)
    invgΣ = similar(g)
    hgΣ = similar(g)
    R´source = similar(R´Z11, size(R´Z11, 1), 0)
    maxiter = 10
    for n in 1:maxiter
        # Σ11 <-- hgh + hg * Σ11 * inv(1 - g*Σ11) * gh
        # Σn1 <-- Σn1 * inv(1 - g*Σ11) * gh
        one!(invgΣ)
        mul!(invgΣ, g, Σ11, -1, 1)
        invgΣgh = ldiv!(lu!(invgΣ), copy!(d.tmp.rr0, gh))
        mul!(hgΣ, hg, Σ11)
        Σn1´ = mul!(Σ11, Σn1, invgΣgh)  # save allocations
        Σ11´ = mul!(Σn1, hgΣ, invgΣgh)  # save allocations
        Σ11´ .+= hgh
        Σn1, Σ11 = Σn1´, Σ11´

        R´φg_candidates = nullspace_qr(Σn1, d.atol)
        # iterate [R´Z11 R´φg_candidates] = [R´source 0] [source_rowspace'; ....] until R´source is full rank
        # Then Σ = [Z21 φgJ_candidates] ([R´Z11 R´φg_candidates] \ R') = [R´Z11 φgJ_candidates] * source_rowspace * inv(R´source) * R'
        # we have built a full-rank basis R´source of the space spanned by [R´Z11 φgJ_candidates], which we can invert
        source_rowspace, R´source, _ = fullrank_decomposition_qr([R´Z11 R´φg_candidates], d.atol)
        # when R´source is square, it will be full rank and invertible.
        size(R´source, 1) == size(R´source, 2) && break
    end
    φgJ_candidates = d.R * Σ11 * R´φg_candidates
    target = [Z21 φgJ_candidates] * source_rowspace
    # R´source is an Adjoint, must covert to do lu! later
    return copy(R´source), target
end

# computes R'*(g0, h₋ g0, g0 h₊, h₋ g0 h₊) R
function integrate_out_bulk!(d::Deflator)
    R = d.R
    g0⁻¹ = copy!(d.tmp.nn, d.ig0)
    # ig0⁻¹R, ig0⁻¹L = g0⁻¹ \ R, g0⁻¹ \ L
    lug0⁻¹ = lu!(g0⁻¹)
    g0R = ldiv!(lug0⁻¹, copy!(d.tmp.nr1, R))
    R´g0R = mul!(d.tmp.rr1, R', g0R)
    h₋g0R = mul!(d.tmp.nr2, d.hm, g0R)
    R´h₋g0R = mul!(d.tmp.rr2, R', h₋g0R)
    h₊R = mul!(d.tmp.nr1, d.hp, R)
    g0h₊R = ldiv!(lug0⁻¹, copy!(d.tmp.nr2, h₊R))
    R´g0h₊R = mul!(d.tmp.rr3, R', g0h₊R)
    h₋g0h₊R = mul!(d.tmp.nr1, d.hm, g0h₊R)
    R´h₋g0h₊R = mul!(d.tmp.rr4, R', h₋g0h₊R)
    return R´g0R, R´h₋g0R, R´g0h₊R, R´h₋g0h₊R
end

### Greens execution

# Choose codepath
function (g::GreensFunction{<:Schur1DGreensSolver})(ω, cells; kw...)
    cells´ = sanitize_cells(cells, g)
    if is_infinite_local(cells´, g)
        gω = local_fastpath(g, ω; kw...)
    elseif is_across_boundary(cells´, g)
        gω = Matrix(zero(g.solver.h0))
    elseif is_at_surface(cells´, g)
        gω = surface_fastpath(g, ω, dist_to_boundary(cells´, g); kw...)
    else # general form
        gω = g(ω, missing; kw...)(cells´)
    end
    gω´ = slice_original_cell(gω, g.solver.unitcells)
    gω´´ = maybeunflatten_blocks(gω´, orbitalstructure(g.h))
    return gω´´
end

slice_original_cell(mat, n) = n === 1 ? mat : mat[1:size(mat,1) ÷ n, 1:size(mat, 2)]

is_infinite_local((src, dst), g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Missing}}) =
    only(src) == only(dst)
is_infinite_local(cells, g) = false

is_at_surface((src, dst), g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Int}}) =
    abs(dist_to_boundary(src, g)) == 1 || abs(dist_to_boundary(dst, g)) == 1
is_at_surface(cells, g) = false

is_across_boundary((src, dst), g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Int}}) =
    sign(dist_to_boundary(src, g)) != sign(dist_to_boundary(src, g)) ||
    dist_to_boundary(src, g) == 0 || dist_to_boundary(dst, g) == 0
is_across_boundary(cells, g) = false

dist_to_boundary(cell, g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Int}}) =
    only(cell) - only(g.boundaries)
dist_to_boundary((src, dst)::Pair, g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Int}}) =
    dist_to_boundary(src, g), dist_to_boundary(dst, g)

## Fast-paths

# Surface-bulk semi-infinite:
# G_{1,1} = (ω*I - h0 - ΣR)⁻¹, G_{-1,-1} = (ω*I - h0 - ΣL)⁻¹
# G_{N,1} = (G_{1,1}h₁)ᴺ⁻¹G_{1,1}, where G_{1,1} = (ω*I - h0 - ΣR)⁻¹
# G_{-N,-1} = (G_{-1,-1}h₋₁)ᴺ⁻¹G_{-1,-1}, where G_{-1,-1} = (ω*I - h0 - ΣL)⁻¹
function surface_fastpath(g, ω, (dsrc, ddst); sources = I)
    dist = ddst - dsrc
    Σ = dsrc > 0 ? g.solver(ω, Val(:R)) : g.solver(ω, Val(:L))
    h0 = g.solver.h0
    sourcemat = flat_padded_source_matrix(sources, g)
    # in-place optimization of luG⁻¹ = lu(ω*I - h0 - Σ)
        G⁻¹ = Σ
        @. G⁻¹ = - h0 - Σ
        shiftω!(G⁻¹, ω)
        luG⁻¹ = lu!(G⁻¹)
    if dist == 0
        G = ldiv!(luG⁻¹, sourcemat)
    else
        h = dist > 0 ? g.solver.hp : g.solver.hm
        Gh = ldiv!(luG⁻¹, Matrix(h))
        G = Gh^(abs(dist)-1) * ldiv!(luG⁻¹, sourcemat)
    end
    return G
end

# Local infinite: G∞_{n,n} = (ω*I - h0 - ΣR - ΣL)⁻¹
function local_fastpath(g, ω; sources = I)
    ΣR, ΣL = g.solver(ω, Val(:RL))
    h0 = g.solver.h0
    sourcemat = flat_padded_source_matrix(sources, g)
    # in-place optimization of luG∞⁻¹ = lu(ω*I - h0 - ΣL - ΣR)
        G∞⁻¹ = ΣR
        @. G∞⁻¹ = - h0 - ΣR - ΣL
        shiftω!(G∞⁻¹, ω)
        luG∞⁻¹ = lu!(G∞⁻¹)
    G∞ = ldiv!(luG∞⁻¹, sourcemat)
    return G∞
end

function flat_padded_source_matrix(::UniformScaling, g)
    n = flatsize(g, 1)
    m = n * g.solver.unitcells
    return Matrix(one(blockeltype(g)) * I, m, n)
end

function flat_padded_source_matrix(sources::Vector{Int}, g)
    n = length(sources)
    m = size(g, 1)
    T = blocktype(g)
    sourcemat = zeros(T, m, n)
    for (i, s) in enumerate(sources)
        sourcemat[s, i] = one(T)
    end
    flatmat = flatten(sourcemat, orbitalstructure(g))
    T´ = eltype(flatmat)
    nu = g.solver.unitcells
    paddedmat = nu == 1 ? flatmat : vcat(flatmat, zeros(T´, (nu-1)*size(flatmat, 1), size(flatmat, 2)))
    return paddedmat
end

## General paths

# Infinite: G∞_{N}  = GVᴺ G∞_{0}
function (g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Missing}})(ω, ::Missing)
    G∞⁻¹, GRh₊, GLh₋ = Gfactors(g.solver, ω)
    luG∞⁻¹ = lu(G∞⁻¹)
    return cells -> G_infinite(luG∞⁻¹, GRh₊, GLh₋, cells)
end

function G_infinite(luG∞⁻¹, GRh₊, GLh₋, (src, dst))
    N = only(dst) - only(src)
    N == 0 && return inv(luG∞⁻¹)
    Ghᴺ = Gh_power(GRh₊, GLh₋, N)
    G∞ = rdiv!(Ghᴺ, luG∞⁻¹)
    return G∞
end

# Semiinifinite: G_{N,M} = (Ghᴺ⁻ᴹ - GhᴺGh⁻ᴹ)G∞_{0}
function (g::GreensFunction{<:Schur1DGreensSolver,1,Tuple{Int}})(ω, ::Missing)
    G∞⁻¹, GRh₊, GLh₋ = Gfactors(g.solver, ω)
    return cells -> G_semiinfinite(G∞⁻¹, GRh₊, GLh₋, dist_to_boundary.(cells, Ref(g)))
end

function G_semiinfinite(G∞⁻¹, Gh₊, Gh₋, (N, M))
    Ghᴺ⁻ᴹ = Gh_power(Gh₊, Gh₋, N-M)
    Ghᴺ = Gh_power(Gh₊, Gh₋, N)
    Gh⁻ᴹ = Gh_power(Gh₊, Gh₋, -M)
    mul!(Ghᴺ⁻ᴹ, Ghᴺ, Gh⁻ᴹ, -1, 1) # (Ghᴺ⁻ᴹ - GhᴺGh⁻ᴹ)
    # G∞ = rdiv!(Ghᴺ⁻ᴹ , luG∞⁻¹)  # This is not defined in Julia (v1.7) yet
    G∞ = Ghᴺ⁻ᴹ / G∞⁻¹
    return G∞
end

function Gfactors(solver::Schur1DGreensSolver, ω)
    ΣR, ΣL = solver(ω, Val(:RL))
    A1 = ω*I - solver.h0
    GR⁻¹ = A1 - ΣR
    GRh₊ = GR⁻¹ \ Matrix(solver.hp)
    GL⁻¹ = A1 - ΣL
    GLh₋ = GL⁻¹ \ Matrix(solver.hm)
    G∞⁻¹ = GL⁻¹ - ΣR
    return G∞⁻¹, GRh₊, GLh₋
end

Gh_power(Gh₊, Gh₋, N) = N == 0 ? one(Gh₊) : N > 0 ? Gh₊^N : Gh₋^-N

#######################################################################
# BandGreensSolver
#######################################################################
# struct SimplexData{D,E,T,C<:SMatrix,DD,SA<:SubArray}
#     ε0::T
#     εmin::T
#     εmax::T
#     k0::SVector{D,T}
#     Δks::SMatrix{D,D,T,DD}     # k - k0 = Δks * z
#     volume::T
#     zvelocity::SVector{D,T}
#     edgecoeffs::NTuple{E,Tuple{T,C}} # s*det(Λ)/w.w and Λc for each edge
#     dωzs::NTuple{E,NTuple{2,SVector{D,T}}}
#     defaultdη::SVector{D,T}
#     φ0::SA
#     φs::NTuple{D,SA}
# end

# struct BandGreensSolver{P<:SimplexData,E,H<:Hamiltonian} <: AbstractGreensSolver
#     simplexdata::Vector{P}
#     indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
#     h::H
# end

# function Base.show(io::IO, g::GreensFunction{<:BandGreensSolver})
#     print(io, summary(g), "\n",
# "  Matrix size    : $(size(g.h, 1)) × $(size(g.h, 2))
#   Element type   : $(displayelements(g.h))
#   Band simplices : $(length(g.solver.simplexdata))")
# end

# Base.summary(g::GreensFunction{<:BandGreensSolver}) =
#     "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.h))D bandstructure"

# function greensolver(b::Bandstructure{D}, h) where {D}
#     indsedges = tuplepairs(Val(D))
#     v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
#     return BandGreensSolver(v,  indsedges, h)
# end

# edges_per_simplex(L) = binomial(L,2)

# function SimplexData(simplex::NTuple{V}, band, indsedges) where {V}
#     D = V - 1
#     vs = ntuple(i -> vertices(band)[simplex[i]], Val(V))
#     ks = ntuple(i -> SVector(Base.front(Tuple(vs[i]))), Val(V))
#     εs = ntuple(i -> last(vs[i]), Val(V))
#     εmin, εmax = extrema(εs)
#     ε0 = first(εs)
#     k0 = first(ks)
#     Δks = hcat(tuple_minus_first(ks)...)
#     zvelocity = SVector(tuple_minus_first(εs))
#     volume = abs(det(Δks))
#     edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
#     dωzs = sectionpoint.(indsedges, Ref(zvelocity))
#     defaultdη = dummydη(zvelocity)
#     φ0 = vertexstate(first(simplex), band)
#     φs = vertexstate.(Base.tail(simplex), Ref(band))
#     return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, defaultdη, φ0, φs)
# end

# function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
#     basis = edgebasis(indsedge, Val(D))
#     othervecs = Base.tail(basis)
#     edgevec = first(basis)
#     cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
#     Λc = hcat(cutvecs...)
#     Λ = hcat(zvelocity, Λc)
#     s = sign(det(hcat(basis...)))
#     coeff = s * (det(Λ)/dot(zvelocity, zvelocity))
#     return coeff, Λc
# end

# function edgebasis(indsedge, ::Val{D}) where {D}
#     inds = ntuple(identity, Val(D+1))
#     swappedinds = tupleswapfront(inds, indsedge) # places the two edge vertindices first
#     zverts = (i->unitvector(SVector{D,Int}, i-1)).(swappedinds)
#     basis = (z -> z - first(zverts)).(Base.tail(zverts)) # first of basis is edge vector
#     return basis
# end

# function sectionpoint((i, j), zvelocity::SVector{D,T}) where {D,T}
#     z0, z1 = unitvector(SVector{D,Int}, i-1), unitvector(SVector{D,Int}, j-1)
#     z10 = z1 - z0
#     # avoid numerical cancellation errors due to zvelocity perpendicular to edge
#     d = chop(dot(zvelocity, z10), maximum(abs.(zvelocity)))
#     dzdω = z10 / d
#     dz0 = z0 - z10 * dot(zvelocity, z0) / d
#     return dzdω, dz0   # The section z is dω * dzdω + dz0
# end

# # A vector, not parallel to zvelocity, and with all nonzero components and none equal
# function dummydη(zvelocity::SVector{D,T}) where {D,T}
#     (D == 1 || iszero(zvelocity)) && return SVector(ntuple(i->T(i), Val(D)))
#     rng = MersenneTwister(0)
#     while true
#         dη = rand(rng, SVector{D,T})
#         isparallel = dot(dη, zvelocity)^2 ≈ dot(zvelocity, zvelocity) * dot(dη, dη)
#         isvalid = allunique(dη) && !isparallel
#         isvalid && return dη
#     end
#     throw(error("Unexpected error finding dummy dη"))
# end

# function vertexstate(ind, band)
#     ϕind = 1 + band.dimstates*(ind - 1)
#     state = view(band.states, ϕind:(ϕind+band.dimstates-1))
#     return state
# end

# ## Call API

# function greens!(matrix, g::GreensFunction{<:BandGreensSolver,L}, ω::Number, (src, dst)::SVectorPair{L}) where {L}
#     fill!(matrix, zero(eltype(matrix)))
#     dn = dst - src
#     for simplexdata in g.solver.simplexdata
#         g0, gjs = green_simplex(ω, dn, simplexdata, g.solver.indsedges)
#         addsimplex!(matrix, g0, gjs, simplexdata)
#     end
#     return matrix
# end

# function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
#     dη = data.Δks' * dn
#     phase = cis(dot(dn, data.k0))
#     dω = ω - data.ε0
#     gz = simplexterm.(dω, Ref(dη), Ref(data), data.edgecoeffs, data.dωzs, indsedges)
#     g0z, gjz = first.(gz), last.(gz)
#     g0 = im^(L-1) * phase * sum(g0z)
#     gj = -im^L * phase * sum(gjz)
#     return g0, gj
# end

# function simplexterm(dω, dη::SVector{D,T}, data, coeffs, (dzdω, dz0), (i, j)) where {D,T}
#     bailout = Complex(zero(T)), Complex.(zero(dη))
#     z = dω * dzdω + dz0
#     # Edges with divergent sections do not contribute
#     all(isfinite, z) || return bailout
#     z0 = unitvector(SVector{D,T},i-1)
#     z1 = unitvector(SVector{D,T},j-1)
#     coeff, Λc = coeffs
#     # If dη is zero (DOS) use a precomputed (non-problematic) simplex-constant vector
#     dη´ = iszero(dη) ? data.defaultdη : dη
#     d = dot(dη´, z)
#     d0 = dot(dη´, z0)
#     d1 = dot(dη´, z1)
#     # Skip if singularity in formula
#     (d ≈ d0 || d ≈ d1) && return bailout
#     s = sign(dot(dη´, dzdω))
#     coeff0 = coeff / prod(Λc' * dη´)
#     coeffj = isempty(Λc) ? zero(dη) : (Λc ./ ((dη´)' * Λc)) * sumvec(Λc)
#     params = s, d, d0, d1
#     zs = z, z0, z1
#     g0z = iszero(dη) ? g0z_asymptotic(D, coeff0, params) : g0z_general(coeff0, params)
#     gjz = iszero(dη) ? gjz_asymptotic(D, g0z, coeffj, coeff0, zs, params) :
#                        gjz_general(g0z, coeffj, coeff0, zs, params)
#     return g0z, gjz
# end

# sumvec(::SMatrix{N,M,T}) where {N,M,T} = SVector(ntuple(_->one(T),Val(M)))

# g0z_general(coeff0, (s, d, d0, d1)) =
#     coeff0 * cis(d) * ((cosint_c(-s*(d0-d)) + im*sinint(d0-d)) - (cosint_c(-s*(d1-d)) + im*sinint(d1-d)))

# gjz_general(g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1)) =
#     g0z * (im * z - coeffj) + coeff0 * ((z0-z) * cis(d0) / (d0-d) - (z1-z) * cis(d1) / (d1-d))

# g0z_asymptotic(D, coeff0, (s, d, d0, d1)) =
#     coeff0 * (cosint_a(-s*(d0-d)) - cosint_a(-s*(d1-d))) * (im*d)^(D-1)/factorial(D-1)

# function gjz_asymptotic(D, g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1))
#     g0z´ = g0z
#     for n in 1:(D-1)
#         g0z´ += coeff0 * im^n * (im*d)^(D-1-n)/factorial(D-1-n) *
#                 ((d0-d)^n - (d1-d)^n)/(n*factorial(n))
#     end
#     gjz = g0z´ * (im * z - im * coeffj * d / D) +
#         coeff0 * ((z0-z) * (im*d0)^D / (d0-d) - (z1-z) * (im*d1)^D / (d1-d)) / factorial(D)
#     return gjz
# end

# cosint_c(x::Real) = ifelse(iszero(abs(x)), zero(x), cosint(abs(x))) + im*pi*(x<=0)

# cosint_a(x::Real) = ifelse(iszero(abs(x)), zero(x), log(abs(x))) + im*pi*(x<=0)

# function addsimplex!(matrix, g0, gjs, simplexdata)
#     φ0 = simplexdata.φ0
#     φs = simplexdata.φs
#     vol = simplexdata.volume
#     for c in CartesianIndices(matrix)
#         (row, col) = Tuple(c)
#         x = g0 * (φ0[row] * φ0[col]')
#         for (φ, gj) in zip(φs, gjs)
#             x += (φ[row]*φ[col]' - φ0[row]*φ0[col]') * gj
#         end
#         matrix[row, col] += vol * x
#     end
#     return matrix
# end