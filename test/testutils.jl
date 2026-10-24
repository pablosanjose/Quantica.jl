# Helpers shared by all test files. Loaded into each test sandbox by runtests.jl

using Quantica: GreenFunction, GreenSlice, GreenSolution, CellOrbitals, ncontacts
using LinearAlgebra: diag

"""
    cycled(xs, i)

Element `i` of `xs`, wrapping around cyclically, i.e. `xs[mod1(i, length(xs))]`.
"""
cycled(xs, i) = xs[mod1(i, length(xs))]

"""
    cyclic_product(full::Tuple, cycling::Tuple)

Return the combinations `(f..., c...)` of all elements of the Cartesian product of the
collections in `full`, each extended with one element `c` from each collection in `cycling`.
The cycling elements are not multiplied in: they are chosen Latin-square style, as
`cycled(c, i₁ + i₂ + ...)` where `iₖ` are the indices into `full`. Hence, every element
of a cycling collection appears together with every element of each `full` collection (provided
the other `full` collections are long enough). So all pairs among `full` collections and all
(full, cycling) pairs are covered, at a fraction of the cost of a full product. Pairs of two
cycling collections are not guaranteed to be fully covered.

# Example
```julia
julia> cyclic_product(((1, 2), (:a, :b, :c)), (("x", "y", "z"),))
6-element Vector{Tuple{Int64, Symbol, String}}:
 (1, :a, "y")
 (2, :a, "z")
 (1, :b, "z")
 (2, :b, "x")
 (1, :c, "x")
 (2, :c, "y")
```
All 6 combinations of `(1, 2)` and `(:a, :b, :c)` appear, instead of the 18 of a full product
with `("x", "y", "z")`. Both `1` and `2` meet all of `"x"`, `"y"`, `"z"`, but each of `:a, :b, :c`
meets only two of them, because `(1, 2)` is too short to cycle through all three.
"""
cyclic_product(full::Tuple, cycling::Tuple) =
    vec([(getindex.(full, Tuple(I))..., (cycled(c, sum(Tuple(I))) for c in cycling)...)
        for I in CartesianIndices(map(eachindex, full))])

"""
    locpairs(locs::Tuple, conts::Tuple; allpairs = false)

Return a vector of `(row, col)` index pairs used by `testgreen` to slice a Green function as
`g[row, col]`, where `locs` are site/orbital selections and `conts` are contact indices.

With `allpairs = true`, return all pairs of `(locs..., conts...)` with itself. Otherwise return
a reduced set:
- each element of `locs` paired with itself and with its two cyclic neighbors in `locs`,
- each element of `locs` paired with each contact, in both orders,
- all pairs of contacts.
Every element thus appears as both row and column, with itself and with others, and every
combination of index kinds (contact, single-cell, multi-cell) is still exercised, with far
fewer pairs to compile (e.g. 41 instead of 81 for 8 `locs` and one contact).

# Example
```julia
julia> locpairs((:a, :b, :c, :d), (1,))
21-element Vector{Tuple{Any, Any}}:
 (:a, :a)
 (:a, :b)
 (:a, :d)
 (:b, :b)
 (:b, :c)
 (:b, :a)
 (:c, :c)
 (:c, :d)
 (:c, :b)
 (:d, :d)
 (:d, :a)
 (:d, :c)
 (:a, 1)
 (1, :a)
 (:b, 1)
 (1, :b)
 (:c, 1)
 (1, :c)
 (:d, 1)
 (1, :d)
 (1, 1)
```
Only `(:a, :c)`, `(:c, :a)`, `(:b, :d)` and `(:d, :b)` are left out of the 25 pairs of the full
product: they are not cyclic neighbors.
"""
function locpairs(locs, conts; allpairs = false)
    allpairs && return vec(collect(Iterators.product((locs..., conts...), (locs..., conts...))))
    n = length(locs)
    neighbors = [(locs[i], locs[mod1(i + o, n)]) for i in 1:n for o in (0, 1, -1)]
    withconts = [p for l in locs for c in conts for p in ((l, c), (c, l))]
    return [neighbors; withconts; vec(collect(Iterators.product(conts, conts)))]
end

function testcond(@nospecialize(g0); nambu = false)
    G1 = conductance(g0[1]; nambu)
    G2 = conductance(g0[2]; nambu)
    G12 = conductance(g0[1,2]; nambu)
    T12 = transmission(g0[1,2])
    @test_throws ArgumentError transmission(g0[1])
    @test_throws ArgumentError transmission(g0[1, 1])
    for ω in -3:0.1:3
        @test ifelse(nambu, 6.000001, 3.00000001) >= G1(ω) >= 0
        @test G1(ω) ≈ G1(-ω) ≈ G2(ω) ≈ G2(-ω)
        @test T12(ω) ≈ T12(-ω)
        nambu || @test G1(ω) ≈ T12(ω) atol = 1e-8
        @test G12(ω) ≈ G12(-ω) atol = 1e-8
        nambu || @test G1(ω) ≈ -G12(ω) atol = 1e-8
    end
end

# @nospecialize: inferring the whole helper body for each concrete h/s type is very costly to compile
function testgreen(@nospecialize(h), @nospecialize(s); allpairs = false, kw...)
    ω = 0.2
    g = greenfunction(h, s)
    @test g isa GreenFunction
    gω = g(ω; kw...)
    @test gω isa GreenSolution
    L = Quantica.latdim(lattice(h))
    z = zero(SVector{L,Int})
    o = Quantica.unitvector(1, SVector{L,Int})
    conts = ntuple(identity, ncontacts(h))
    locs = (sites(z, :), sites(z, 2), sites(1:2), sites(o, (1,2)), (; cells = z),
        CellOrbitals(o, 1), CellOrbitals(z, :), CellOrbitals(z, SA[1,2]))
    for (loc, loc´) in locpairs(locs, conts; allpairs)
        gs = g[loc, loc´]
        @test gs isa GreenSlice
        gsω = gs(ω; kw...)
        gωs = gω[loc, loc´]
        @test gsω == gωs
        loc === loc´ && @test all(x->imag(x)<=0, diag(gωs))
    end
    return nothing
end

# @nospecialize: whole-body inference over all the Josephson types takes minutes to compile otherwise
function testjosephson(@nospecialize(g0))
    J1 = josephson(g0[1], 1; phases = subdiv(0, pi, 10))
    J2 = josephson(g0[2], 1; phases = subdiv(0, pi, 10))
    J3 = josephson(g0[1], (-5, 0); phases = subdiv(0, pi, 10))
    J4 = josephson(g0[1], Paths.sawtooth(range(-5, 0, 3)); phases = subdiv(0, pi, 10))
    p5(µ, T) = ifelse(iszero(T),(-5,-2+3im,0),(-5,-2+3im,0,2+3im,5)) .+ sqrt(eps(Float64))*im
    J5 = josephson(g0[1], Paths.polygon(p5); phases = subdiv(0, pi, 10))
    j1 = J1()
    @test all(>=(0), Quantica.chopsmall.(j1))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J2()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J3()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J4()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(j1, J5()))
    @test all(((j1, j2) -> ≈(j1, j2, atol = 1e-6)).(J1(0.1), J5(0.1)))

    j = Quantica.integrand(J1)
    @test Quantica.call!(j, 0.2) isa Vector
    # static length of integration points (required for QuadGK.quadgk type-stability)
    @test typeof.(Quantica.points.((J1, J2, J3, J4, J5))) ==
        (Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}, Tuple{ComplexF64, ComplexF64, ComplexF64})
    # integration path logic
    J = josephson(g0[1], Paths.sawtooth(-4, -1); callback = ((x,y) -> @test x isa ComplexF64 && y isa ComplexF64))
    @test J() <= 10*eps(Float64)
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3    # tuple sawtooth
    @test p2 isa Vector{ComplexF64} && length(p2) == 3
    J = josephson(g0[1], Paths.sawtooth(-4, 0); phases = subdiv(0, pi, 10))
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3    # tuple sawtooth
    @test p2 isa Vector{ComplexF64} && length(p2) == 3
    @test all(p2 .≈ p1)
    J = josephson(g0[1], Paths.sawtooth(-4, 1); phases = subdiv(0, pi, 10))
    p1, p2 = Quantica.points(J, 0), Quantica.points(J, 0.2)
    @test p1 isa Vector{ComplexF64} && length(p1) == 3 && maximum(real(p1)) == 0
    @test p2 isa Vector{ComplexF64} && length(p2) == 5 && maximum(real(p2)) == 1
end
