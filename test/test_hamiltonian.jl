using LinearAlgebra: diag, norm, det
using Quantica: Hamiltonian, ParametricHamiltonian, nhoppings, nonsites, nsites, coordination, allsitepositions

@testset "basic hamiltonians" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc, LatticePresets.hcp)
    types = (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
    ts = (1, 2.0I, @SMatrix[1 2; 3 4], 1.0f0*I)
    orbs = (Val(1), Val(1), Val(2), (Val(1), Val(2)))
    for preset in presets, lat in (preset(), unitcell(preset())), type in types
        E, L = dims(lat)
        dn0 = ntuple(_ -> 1, Val(L))
        for (t, o) in zip(ts, orbs)
            @test hamiltonian(lat, onsite(t) + hopping(t; range = 1), orbitals = o, orbtype = type) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) - hopping(t; dn = dn0), orbitals = o, orbtype = type) isa Hamiltonian
        end
    end
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3))
    @test bloch(h) == h.harmonics[1].h
    # Inf range
    h = LatticePresets.square() |> unitcell(region = RegionPresets.square(5)) |>
        hamiltonian(hopping(1, range = Inf))
    @test nhoppings(h) == 600
    h = LatticePresets.square() |> hamiltonian(hopping(1, dn = (10,0), range = Inf))
    @test nhoppings(h) == 1
    @test isassigned(h, (10,0))
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(1.0, sublats = :A), orbitals = (Val(1), Val(2)))
    @test Quantica.nonsites(h) == 1
    h = LatticePresets.square() |> unitcell(3) |> hamiltonian(hopping(1, indices = (1:8 .=> 2:9, 9=>1), range = 3, plusadjoint = true))
    @test nhoppings(h) == 48
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 1)))
    @test nhoppings(h) == 12
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (1, 2/√3)))
    @test nhoppings(h) == 18
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (2, 1)))
    @test Quantica.nhoppings(h) == 0
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (30, 30)))
    @test Quantica.nhoppings(h) == 12
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = (10, 10.1)))
    @test Quantica.nhoppings(h) == 48
end

@testset "hamiltonian unitcell" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r -> abs(r[2])<2)
    @test nhoppings(h) == 22
    h = LatticePresets.square() |> hamiltonian(hopping(1)) |> unitcell(3) |> unitcell((1,0), indices = not(1))
    @test nsites(h) == 8
    h = LatticePresets.square() |> hamiltonian(hopping(1, range = √2)) |> unitcell(5) |> unitcell((1,0), indices = 1:2:25)
    @test nhoppings(h) == 38
    @test nsites(h) == 13
    h = LatticePresets.square() |> hamiltonian(hopping(1, range = √2)) |> unitcell(5) |> unitcell(indices = 2:2:25)
    @test nhoppings(h) == 32
    @test nsites(h) == 12
    lat = LatticePresets.honeycomb(dim = Val(3)) |> unitcell(3) |> unitcell((1,1), indices = not(1))
    h = lat |> hamiltonian(hopping(1)) |> unitcell
    @test nsites(h) == 17
    h = unitcell(h, indices = not(1))
    @test nsites(h) == 16
    # dim-preserving unitcell reshaping should always preserve coordination
    lat = lattice(sublat((0.0, -0.1), (0,0), (0,1/√3), (0.0,1.6/√3)), bravais = SA[cos(pi/3) sin(pi/3); -cos(pi/3) sin(pi/3)]')
    h = lat |> hamiltonian(hopping(1, range = 1/√3))
    c = coordination(h)
    iter = CartesianIndices((-3:3, -3:3))
    for Ic´ in iter, Ic in iter
        sc = SMatrix{2,2,Int}(Tuple(Ic)..., Tuple(Ic´)...)
        iszero(det(sc)) && continue
        h´ = unitcell(h, sc)
        h´´ = unitcell(h´, 2)
        @test coordination(h´´) ≈ coordination(h´) ≈ c
        h´ = unitcell(h´, Tuple(Ic))
        h´´ = unitcell(h´, 2)
        @test coordination(h´´) ≈ coordination(h´)
    end
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell(2) |> unitcell(mincoordination = 2)
    @test nsites(h) == 6
    h = LP.cubic() |> hamiltonian(hopping(1)) |> unitcell(4) |> unitcell(mincoordination = 4)
    @test nsites(h) == 0
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell(region = RP.circle(5) & !RP.circle(2)) |> unitcell(mincoordination = 2)
    @test nsites(h) == 144
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell(10, region = !RP.circle(2, (0,8)))
    h´ = h |> unitcell(1, mincoordination = 2)
    @test nsites(h´) == nsites(h) - 1
    h = LP.square() |> hamiltonian(hopping(0)) |> unitcell(4, mincoordination = 2)
    @test nsites(h) == 0
    # check dn=0 invariant
    h = LP.linear() |> hamiltonian(hopping(1)) |> unitcell((1,))
    @test length(h.harmonics) == 3 && iszero(first(h.harmonics).dn)
end

@testset "hamiltonian wrap" begin
    h = LatticePresets.bcc() |> hamiltonian(hopping((r, dr) -> 1/norm(dr), range = 10))
    wh = wrap(h, phases = (1,2,3))
    @test bloch(wh) ≈ bloch(h, (1,2,3))
    h = LatticePresets.bcc() |> hamiltonian(hopping((r, dr) -> 1/norm(dr), range = 10)) |> unitcell(3)
    wh = wrap(h, phases = (1,2,3))
    @test bloch(wh) ≈ bloch(h, (1,2,3))
end

@testset "similarmatrix" begin
    types = (ComplexF16, ComplexF32, ComplexF64)
    lat = LatticePresets.honeycomb()
    for T in types
        h0 = hamiltonian(lat, onsite(I) + hopping(2I; range = 1), orbitals = (Val(1), Val(2)), orbtype = T)
        hf = flatten(h0)
        hm = Matrix(h0)
        hs = (h0, hf, hm)
        As = (SparseMatrixCSC, SparseMatrixCSC, Matrix)
        Es = (SMatrix{2,2,T,4}, T, SMatrix{2,2,T,4})
        for (h, A, E) in zip(hs, As, Es)
            sh = similarmatrix(h)
            @test sh isa A{E}
            b1 = bloch!(similarmatrix(flatten(h)), flatten(h), (1,1))
            b2 = bloch!(similarmatrix(h, flatten), h, (1,1))
            @test isapprox(b1, b2)
            for T´ in types
                E´s = E <: SMatrix ? (SMatrix{2,2,T´,4}, T´) : (T´,)
                for E´ in E´s
                    s1 = similarmatrix(h, Matrix{E´})
                    s2 = similarmatrix(h, Matrix)
                    @test s1 isa Matrix{E´}
                    @test s2 isa Matrix{E}
                    if A != Matrix
                        s1 = similarmatrix(h, SparseMatrixCSC{E´})
                        s2 = similarmatrix(h, SparseMatrixCSC)
                        @test s1 isa SparseMatrixCSC{E´}
                        @test s2 isa SparseMatrixCSC{E}
                    end
                end
            end
        end
    end

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(2)))
    s = similarmatrix(h)
    @test size(s) == (2,2) && s isa SparseMatrixCSC{<:SMatrix{2,2}}
    s = similarmatrix(h, flatten)
    @test size(s) == (3,3) && s isa SparseMatrixCSC{ComplexF64}
    s = similarmatrix(h, SparseMatrixCSC)
    @test size(s) == (2,2) && s isa SparseMatrixCSC{<:SMatrix{2,2}}
    s = similarmatrix(h, SparseMatrixCSC{ComplexF16})
    @test size(s) == (3,3) && s isa SparseMatrixCSC{ComplexF16}
    s = similarmatrix(h, Matrix)
    @test size(s) == (2,2) && s isa Matrix{<:SMatrix{2,2}}
    s = similarmatrix(h, Matrix{Float64})
    @test size(s) == (3,3) && s isa Matrix{Float64}

    h = Matrix(h)
    s = similarmatrix(h)
    @test size(s) == (2,2) && s isa Matrix{<:SMatrix{2,2}}
    s = similarmatrix(h, flatten)
    @test size(s) == (3,3) && s isa Matrix{ComplexF64}
    @test_throws ArgumentError similarmatrix(h, SparseMatrixCSC)
    @test_throws ArgumentError similarmatrix(h, SparseMatrixCSC{ComplexF16})
    s = similarmatrix(h, Matrix)
    @test size(s) == (2,2) && s isa Matrix{<:SMatrix{2,2}}
    s = similarmatrix(h, Matrix{Float64})
    @test size(s) == (3,3) && s isa Matrix{Float64}

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I))
    s = similarmatrix(h)
    @test size(s) == (2,2) && s isa SparseMatrixCSC{ComplexF64}
    s = similarmatrix(h, flatten)
    @test size(s) == (2,2) && s isa SparseMatrixCSC{ComplexF64}
    s = similarmatrix(h, SparseMatrixCSC)
    @test size(s) == (2,2) && s isa SparseMatrixCSC{ComplexF64}
    s = similarmatrix(h, SparseMatrixCSC{ComplexF16})
    @test size(s) == (2,2) && s isa SparseMatrixCSC{ComplexF16}
    s = similarmatrix(h, Matrix)
    @test size(s) == (2,2) && s isa Matrix{ComplexF64}
    s = similarmatrix(h, Matrix{Float64})
    @test size(s) == (2,2) && s isa Matrix{Float64}

    h = Matrix(h)
    s = similarmatrix(h)
    @test size(s) == (2,2) && s isa Matrix{ComplexF64}
    s = similarmatrix(h, flatten)
    @test size(s) == (2,2) && s isa Matrix{ComplexF64}
    @test_throws ArgumentError similarmatrix(h, SparseMatrixCSC)
    @test_throws ArgumentError similarmatrix(h, SparseMatrixCSC{ComplexF16})
    s = similarmatrix(h, Matrix)
    @test size(s) == (2,2) && s isa Matrix{ComplexF64}
    s = similarmatrix(h, Matrix{Float64})
    @test size(s) == (2,2) && s isa Matrix{Float64}
end

@testset "orbitals and sublats" begin
    orbs = (:a, (:a,), (:a, :b, 3), ((:a, :b), :c), ((:a, :b), (:c,)), (Val(2), Val(1)),
            (:A => (:a, :b), :D => :c), :D => Val(4))
    lat = LatticePresets.honeycomb()
    for orb in orbs
        @test hamiltonian(lat, onsite(I), orbitals = orb) isa Hamiltonian
    end
    @test hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = :A =>:B),
                      orbitals = :A => Val(2)) isa Hamiltonian
    h1 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = :A =>:B),
                      orbitals = :A => Val(2))
    h2 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = (:A =>:B,)),
                      orbitals = :A => Val(2))
    @test bloch(h1, (1, 2)) == bloch(h2, (1, 2))
end

@testset "onsite dimensions" begin
    lat = LatticePresets.honeycomb()
    ts = (@SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4], 2, r -> 2r[1],
          @SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4])
    os = (Val(1), Val(1), Val(2), Val(2), Val(3), Val(3))
    for (t, o) in zip(ts, os)
        model = onsite(t)
        @test_throws DimensionMismatch hamiltonian(lat, model, orbitals = o)
    end

    ts = (@SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4], 2, r -> 2r[1], 3I, r -> I, 3I)
    os = (Val(2), Val(2), Val(1), Val(1), Val(3), Val(3), (Val(1), Val(3)))
    for (t, o) in zip(ts, os)
        model = onsite(t)
        @test hamiltonian(lat, model, orbitals = o) isa Hamiltonian
    end
    @test bloch(hamiltonian(lat, onsite(3I), orbitals = (Val(1), Val(3))))[1,1] ==
        @SMatrix[3 0 0; 0 0 0; 0 0 0]
    @test bloch(hamiltonian(lat, onsite(3I), orbitals = (Val(1), Val(3))))[2,2] ==
        SMatrix{3,3}(3I)
end

@testset "hopping dimensions" begin
    lat = LatticePresets.honeycomb()
    ts = (@SMatrix[1 2; 2 3], (r,dr) -> @SMatrix[r[1] 2; 3 4], 2, (r,dr) -> 2r[1],
          @SMatrix[1 2], @SMatrix[1 ;2], @SMatrix[1 2], @SMatrix[1 2; 2 3])
    os = (Val(1), Val(1), Val(2), Val(2), (Val(2), Val(1)), (Val(2), Val(1)),
         (Val(2), Val(1)), (Val(2), Val(1)))
    ss = (missing, missing, missing, missing, :B => :A, :A => :B, missing, missing)
    for (t, o, s) in zip(ts, os, ss)
        model = hopping(t, sublats = s)
        @test_throws DimensionMismatch hamiltonian(lat, model, orbitals = o)
    end
    ts = (@SMatrix[1 2], @SMatrix[1 ;2])
    os = ((Val(2), Val(1)), (Val(2), Val(1)))
    ss = (:A => :B, :B => :A)
    for (t, o, s) in zip(ts, os, ss)
        model = hopping(t, sublats = s)
        @test hamiltonian(lat, model, orbitals = o) isa Hamiltonian
    end
    @test bloch(hamiltonian(lat, hopping(3I, range = 1/√3), orbitals = (Val(1), Val(2))))[2,1] ==
        @SMatrix[3 0; 0 0]
    @test bloch(hamiltonian(lat, hopping(3I, range = 1/√3), orbitals = (Val(1), Val(2))))[1,2] ==
        @SMatrix[3 0; 0 0]
end

@testset "hermiticity" begin
    lat = LatticePresets.honeycomb()
    @test !ishermitian(hamiltonian(lat, hopping(im, sublats = :A=>:B)))
    @test !ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B)))
    @test !ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, dn = (-1,0))))
    @test !ishermitian(hamiltonian(lat, hopping(im)))
    @test ishermitian(hamiltonian(lat, hopping(1)))

    @test ishermitian(hamiltonian(lat, hopping(im, sublats = :A=>:B, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, dn = (1,0), plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(im, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, plusadjoint = true)))
end

@testset "unitcell modifiers" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(0)) |> unitcell(2, modifiers = (@onsite!((o, r) -> 1), @hopping!(h -> 1)))
    @test diag(bloch(h)) == ComplexF64[1, 1, 1, 1, 1, 1, 1, 1]
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(0)) |> unitcell(2) |> unitcell(2, modifiers = @onsite!(o -> 1; indices = 3), indices = not(2))
    @test nonsites(h) == 4
    @test nsites(h) == 28
end

@testset "@onsite!" begin
    el = @SMatrix[1 2; 2 1]

    @test @onsite!(o -> 2o)(el) == 2el
    @test @onsite!(o -> 2o)(el, p = 2) == 2el
    @test @onsite!((o;) -> 2o)(el) == 2el
    @test @onsite!((o;) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z) -> 2o)(el, z = 2) == 2el
    @test @onsite!((o; z) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, z = 1, p = 2) == 2el
    @test @onsite!((o; kw...) -> 2o)(el) == 2el
    @test @onsite!((o; kw...) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z, kw...) -> 2o)(el, z = 2) == 2el
    @test @onsite!((o; z, kw...) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z, y = 2, kw...) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z, y = 2, kw...) -> 2o)(el, z = 2, y = 3, p = 2) == 2el

    r = SVector(0,0)

    @test @onsite!((o, r;) -> 2o)(el, r) == 2el
    @test @onsite!((o, r;) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o, r; z) -> 2o)(el, r, z = 2) == 2el
    @test @onsite!((o, r; z) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r, z = 1, p = 2) == 2el
    @test @onsite!((o, r; kw...) -> 2o)(el, r) == 2el
    @test @onsite!((o, r; kw...) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o, r; z, kw...) -> 2o)(el, r, z = 2) == 2el
    @test @onsite!((o, r; z, kw...) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o)(el, r, z = 2, y = 3, p = 2) == 2el

    @test @onsite!((o; z, y = 2, kw...) -> 2o) isa Quantica.UniformOnsiteModifier
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o) isa Quantica.OnsiteModifier{2}

    @test parameters(@onsite!((o, r; z, y = 2, kw...) -> 2o)) == (:z, :y)
end

@testset "@hopping!" begin
    el = @SMatrix[1 2; 2 1]

    @test @hopping!(t -> 2t)(el) == 2el
    @test @hopping!(t -> 2t)(el, p = 2) == 2el
    @test @hopping!((t;) -> 2t)(el) == 2el
    @test @hopping!((t;) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z) -> 2t)(el, z = 2) == 2el
    @test @hopping!((t; z) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, z = 1, p = 2) == 2el
    @test @hopping!((t; kw...) -> 2t)(el) == 2el
    @test @hopping!((t; kw...) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z, kw...) -> 2t)(el, z = 2) == 2el
    @test @hopping!((t; z, kw...) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z, y = 2, kw...) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z, y = 2, kw...) -> 2t)(el, z = 2, y = 3, p = 2) == 2el

    r = dr = SVector(0,0)

    @test @hopping!((t, r, dr;) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t, r, dr;) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t, r, dr; z) -> 2t)(el, r, dr, z = 2) == 2el
    @test @hopping!((t, r, dr; z) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr, z = 1, p = 2) == 2el
    @test @hopping!((t, r, dr; kw...) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t, r, dr; kw...) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t, r, dr; z, kw...) -> 2t)(el, r, dr, z = 2) == 2el
    @test @hopping!((t, r, dr; z, kw...) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t)(el, r, dr, z = 2, y = 3, p = 2) == 2el

    @test @hopping!((t; z, y = 2, kw...) -> 2t) isa Quantica.UniformHoppingModifier
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t) isa Quantica.HoppingModifier{3}

    @test parameters(@hopping!((o, r, dr; z, y = 2, kw...) -> 2o)) == (:z, :y)
end

@testset "parametric" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |> unitcell(10)
    T = typeof(h)
    @test parametric(h, @onsite!(o -> 2o))() isa T
    @test parametric(h, @onsite!((o, r) -> 2o))() isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o))() isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o))(a=1) isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o), @hopping!(t -> 2t))(a=1) isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr) -> r[1]*t))() isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))() isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))(a=1) isa T
    @test parametric(h, @onsite!((o, r; b) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))(b=1) isa T
    @test parametric(h, @onsite!((o, r; b) -> o*b), @hopping!((t, r, dr; a = 2) -> r[1]*t))(a=1, b=2) isa T

    ph = LatticePresets.linear() |> hamiltonian(hopping(1)) |> parametric(@onsite!((o; k) -> o + k*I))
    # No onsites, no need to specify k
    @test ph() isa Hamiltonian
    ph = LatticePresets.linear() |> hamiltonian(onsite(0)) |> parametric(@onsite!((o; k) -> o + k*I))
    ph´ = LatticePresets.linear() |> hamiltonian(onsite(0)) |> parametric(@onsite!((o; k = 1) -> o + k*I))
    @test_throws UndefKeywordError ph()
    @test bloch(ph, (;k = 1)) == bloch(ph(k = 1)) == bloch(ph´)

    # Issue #35
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> unitcell(10)
        ph = parametric(h, @onsite!((o, r; b) -> o+b*I), @hopping!((t, r, dr; a = 2) -> t+r[1]*I),
                       @onsite!((o, r; b) -> o-b*I), @hopping!((t, r, dr; a = 2) -> t-r[1]*I))
        @test isapprox(bloch(ph(a=1, b=2), (1, 2)), bloch(h, (1, 2)))
    end
    # Issue #37
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> unitcell(10)
        ph = parametric(h, @onsite!(o -> o*cis(1)))
        @test ph()[1,1] ≈ h[1,1]*cis(1)
    end
    # Issue #54. Parametric Haldane model
    sK(dr::SVector) = sK(atan(dr[2],dr[1]))
    sK(ϕ) = 2*mod(round(Int, 6*ϕ/(2π)), 2) - 1
    ph = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1)) |>
         parametric(@hopping!((t, r, dr; λ) ->  λ*im*sK(dr); sublats = :A=>:A),
                    @hopping!((t, r, dr; λ) -> -λ*im*sK(dr); sublats = :B=>:B))
    @test bloch(ph(λ=1), (π/2, -π/2)) == bloch(ph, (π/2, -π/2, (;λ=1))) ≈ [4 1; 1 -4]
    # Non-numeric parameters
    ph = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1)) |>
         parametric(@hopping!((t, r, dr; λ, k) ->  λ*im*sK(dr+k); sublats = :A=>:A),
                    @hopping!((t, r, dr; λ, k) -> -λ*im*sK(dr+k); sublats = :B=>:B))
    @test bloch(ph(λ=1, k=SA[1,0]), (π/2, -π/2)) == bloch(ph, (π/2, -π/2, (;λ = 1, k = SA[1,0]))) ≈ [-4 1; 1 4]
    # Issue 61, type stability
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(0))
    @inferred parametric(h, @onsite!((o;μ) -> o- μ))
    @inferred parametric(h, @onsite!(o->2o), @hopping!((t)->2t), @onsite!((o, r)->o+r[1]))
    @inferred parametric(h, @onsite!((o, r)->o*r[1]), @hopping!((t; p)->p*t), @onsite!((o; μ)->o-μ))

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(2I), orbitals = (Val(2), Val(1)))
    ph = parametric(h, @hopping!((t; α, β = 0) -> α * t .+ β))
    b = bloch!(similarmatrix(ph, flatten), ph, (0, 0, (; α = 2)))
    @test b == [0 0 12; 0 0 0; 12 0 0]

    # @block! modifiers
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(0I) + hopping(I), orbitals = (Val(1), Val(3))) |> unitcell(4)
    sites = siteindices(h, region = r->r[1]<0);
    ph = h |> parametric(@onsite!((o; λ=0) -> o - λ*I), @block!((b; λ=0) -> b + λ*I, sites))
    @test_throws ArgumentError ph(λ = 1)
    ph = h |> parametric(@onsite!((o; λ=0) -> o - λ*I), @block!((b; λ=0) -> b + λ*I, sites); check = false)
    @test ph(λ = 1) isa Quantica.Hamiltonian
    sites = siteindices(h, region = r->r[1]<0, sublats = :B);
    ph = h |> parametric(@onsite!((o; λ=0) -> o - λ*I; sublats = :B), @block!((b; λ=0) -> b + λ*I, sites))
    h3 = ph(λ = 3)
    for i in Quantica.siterange(h.lattice, 1)
        @test iszero(h3[i, i])
    end
    for i in Quantica.siterange(h.lattice, 2)
        @test i in sites ? iszero(h3[i,i]) : h3[i,i] == -3I
    end

    h = LatticePresets.square() |> hamiltonian(onsite(0I) + hopping(I)) |> unitcell(4)
    rows = siteindices(h, region = r->r[1]==0);
    cols = siteindices(h, region = r->r[1]==3);
    ph = h |> parametric(
        @block!((b; λ=0) -> b + λ*I, rows, cols; dn = ((0,0),(1,0))),
        @block!((b; λ=0) -> b + λ*I, cols, rows; dn = (-1,0)))
    @test ph(λ = 10)[(1,0)][collect(rows), collect(cols)] == 11I
    @test ph(λ = 10)[(-1,0)][collect(cols), collect(rows)] == 11I
    # check that reset works by calling twice
    @test ph(λ = 10)[(1,0)][collect(rows), collect(cols)] == 11I
    @test ph(λ = 10)[(-1,0)][collect(cols), collect(rows)] == 11I
end

@testset "boolean masks" begin
    for b in ((), (1,1), 4)
        h1 = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |>
             supercell(b, region = RegionPresets.circle(10))
        h2 = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |>
             supercell(b, region = RegionPresets.circle(20))

        @test isequal(h1 & h2, h1)
        @test isequal(h1, h2) || !isequal(h1 & h2, h2)
        @test isequal(h1, h2) || !isequal(h1 | h2, h1)
        @test  isequal(h1 | h2, h2)

        @test isequal(unitcell(h1 & h2), unitcell(h1))
        @test isequal(h1, h2) || !isequal(unitcell(h1 & h2), unitcell(h2))
        @test isequal(h1, h2) || !isequal(unitcell(h1 | h2), unitcell(h1))
        @test isequal(unitcell(h1 | h2), unitcell(h2))

        h1 = h1.lattice
        h2 = h2.lattice

        @test isequal(h1 & h2, h1)
        @test isequal(h1, h2) || !isequal(h1 & h2, h2)
        @test isequal(h1, h2) || !isequal(h1 | h2, h1)
        @test  isequal(h1 | h2, h2)

        @test isequal(unitcell(h1 & h2), unitcell(h1))
        @test isequal(h1, h2) || !isequal(unitcell(h1 & h2), unitcell(h2))
        @test isequal(h1, h2) || !isequal(unitcell(h1 | h2), unitcell(h1))
        @test isequal(unitcell(h1 | h2), unitcell(h2))
    end
end

@testset "unitcell seeds" begin
    p1 = SA[100,0]
    p2 = SA[0,20]
    lat = LatticePresets.honeycomb()
    model = hopping(1, range = 1/√3) + onsite(2)
    h1 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p1)<3, seed = p1)
    h2 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p2)<3, seed = p2)
    h = unitcell(h1 | h2)
    h3 = lat |> hamiltonian(model) |> unitcell(region = r -> norm(r-p1)<3 || norm(r-p2)<3, seed = p2)

    @test Quantica.nsites(h) == 130
    @test Quantica.nsites(h3) == 64
end

@testset "nrange" begin
    lat = LatticePresets.honeycomb(a0 = 2)
    @test nrange(1, lat) ≈ 2/√3
    @test nrange(2, lat) ≈ 2
    @test nrange(3, lat) ≈ 4/√3
    @test hamiltonian(lat, hopping(1)) |> nhoppings == 6
    @test hamiltonian(lat, hopping(1, range = nrange(2))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = nrange(3))) |> nhoppings == 24
    @test hamiltonian(lat, hopping(1, range = (nrange(2), nrange(3)))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = (nrange(20), nrange(20)))) |> nhoppings == 18
    @test hamiltonian(lat, hopping(1, range = (nrange(20), nrange(21)))) |> nhoppings == 30
end

@testset "hamiltonian algebra" begin
    h1 = LatticePresets.honeycomb() |> hamiltonian(onsite(-I) + hopping(I), orbitals = (Val(2), Val(1))) |> unitcell(4)
    h2 = LatticePresets.honeycomb() |> hamiltonian(hopping(3I, range = 1), orbitals = (Val(2), Val(1))) |> unitcell(4)
    phis = (0.3, 0.8)
    @test bloch(3.3*h1, phis) ≈ 3.3*bloch(h1, phis)
    @test bloch(3.3*h1 + h2, phis) ≈ 3.3*bloch(h1, phis) + bloch(h2, phis)
    @test bloch(h1^2, phis) ≈ bloch(h1, phis)^2
    h´ = 2.3h1^3 - 2h2*h1 - 3I
    @test Quantica.check_orbital_consistency(h´) === nothing
    @test bloch(flatten(h´), phis) ≈ 2.3bloch(flatten(h1), phis)^3 - 2bloch(flatten(h2), phis)*bloch(flatten(h1), phis) - 3I
end

@testset "transform! hamiltonians" begin
    h = LP.honeycomb(dim = 3) |> hamiltonian(hopping(1))
    h1 = copy(h)
    h2 = transform!(r -> SA[1 2 0; 2 3 0; 0 0 1] * r + SA[0,0,1], h1)
    h3 = h1 |> transform!(r -> SA[1 2 0; 2 3 0; 0 0 1] * r + SA[0,0,1])
    @test h1 === h2 === h3
    @test all(r->r[3] == 2.0, allsitepositions(h3.lattice))
    @test bloch(h, (1,2)) == bloch(h3, (1,2))
end

@testset "combine hamiltonians" begin
    h1 = LP.square(dim = Val(3)) |> hamiltonian(hopping(1))
    h2 = transform!(r -> r + SA[0,0,1], copy(h1))
    h = combine(h1, h2; coupling = hopping((r,dr) -> exp(-norm(dr)), range = √2))
    @test coordination(h) == 9
    h0 = LP.honeycomb(dim = Val(3), names = (:A, :B)) |> hamiltonian(hopping(1))
    ht = LP.honeycomb(dim = Val(3), names = (:At, :Bt)) |> hamiltonian(hopping(1)) |> transform!(r -> r + SA[0,1/√3,1])
    hb = LP.honeycomb(dim = Val(3), names = (:Ab, :Bb)) |> hamiltonian(hopping(1)) |> transform!(r -> r + SA[0,1/√3,-1])
    h = combine(hb, h0, ht; coupling = hopping((r,dr) -> exp(-norm(dr)), range = 2,
        sublats = ((:A,:B) => (:At, :Bt), (:A,:B) => (:Ab, :Bb)),  plusadjoint = true))
    @test iszero(bloch(h)[1:2, 5:6])
    h = combine(hb, h0, ht; coupling = hopping((r,dr) -> exp(-norm(dr)), range = 2))
    @test !iszero(bloch(h)[1:2, 5:6])
end