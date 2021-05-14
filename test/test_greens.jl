using LinearAlgebra: tr

@testset "greens singleshot selfenergy" begin
    h1 = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    h2 = LP.honeycomb() |> hamiltonian(hopping(1) + onsite(0.02, sublats = :A) - onsite(0.02, sublats = :B)) |> unitcell((4,4), region = r -> -5 < r[1] < 5)
    res(g, ω) = (s = g.solver; Σ = s(ω); sum(abs.(Σ - s.hm * (((ω * I - s.h0) - Σ) \ Matrix(s.hp)))))
    for h in (h1, h2)
        g = greens(h, Schur1D(), boundaries = (0,))
        residual = maximum(res(g, ω) for ω in range(-3.2, 3.2, length = 101))
        g = greens(h, Schur1D(deflation = nothing), boundaries = (0,))
        residual0 = maximum(res(g, ω) for ω in range(-3.2, 3.2, length = 101))
        @test residual < 2*residual0
    end
end

@testset "greens multiorbital" begin
    h = LP.honeycomb() |> hamiltonian(hopping(SA[0 1; 1 0], range = 2), orbitals = Val(2)) |> unitcell((2,-1), region = r->abs(r[2])<3)
    g = greens(h, Schur1D(), boundaries = (0,))
    g0 = greens(flatten(h), Schur1D(), boundaries = (0,))
    @test tr(tr(g(0.2))) ≈ tr(g0(0.2))
end

@testset "greens singleshot spectra" begin
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    g = greens(h, Schur1D())
    dos = [-imag(tr(g(w))) for w in range(-3.2, 3.2, length = 101)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    h = LP.honeycomb() |> hamiltonian(hopping(1, range = 2)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    g = greens(h, Schur1D())
    dos = [-imag(tr(g(w))) for w in range(-6, 6, length = 101)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(2,2)) |> wrap(2)
    g = greens(h, Schur1D())
    dos = [-imag(tr(g(w))) for w in range(-3.2, 3.2, length = 101)]
    @test all(x -> Quantica.chop(x) >= 0, dos)

    # Broken until https://github.com/Reference-LAPACK/lapack/issues/477 comes to julia
    if VERSION >= v"1.7.0-DEV"
        h = LatticePresets.honeycomb() |> hamiltonian(hopping((r, dr) -> 1/(dr'*dr), range = 1)) |> unitcell((1,-1),(2,2)) |> wrap(2)
        g = greens(h, Schur1D())
        dos = [-imag(tr(g(w))) for w in range(-3.2, 3.2, length = 101)]
        @test all(x -> Quantica.chop(x) >= 0, dos)
    end

    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((2,-2),(3,3)) |> unitcell(1, 1, indices = not(1)) |> wrap(2)
    g = greens(h, Schur1D())
    g = greens(h, Schur1D())
    @test Quantica.chop(imag(tr(g(0.2)))) <= 0
    dos = [-imag(tr(g(w))) for w in range(-3.2, 3.2, length = 101)]
    @test all(x -> Quantica.chop(x) >= 0, dos)
end

@testset "greens singleshot spatial" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(3,3)) |> unitcell((1,0))
    g = greens(h, Schur1D(), boundaries = (0,))
    g0 = g(0.01, missing)
    ldos = [-imag(tr(g0(n=>n))) for n in 1:200]
    @test all(x -> Quantica.chop(x) >= 0, ldos)
end

