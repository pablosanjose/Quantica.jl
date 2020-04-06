using LinearAlgebra: diag
using Quantica: Hamiltonian

@testset "basic hamiltonians" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    ts = (1, 2.0I, @SMatrix[1 2; 3 4])
    orbs = (Val(1), Val(1), Val(2))
    for preset in presets, lat in (preset(), unitcell(preset()))
        E, L = dims(lat)
        dn0 = ntuple(_ -> 1, Val(L))
        for (t, o) in zip(ts, orbs)
            @test hamiltonian(lat, onsite(t) + hopping(t; range = 1), orbitals = o) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) - hopping(t; dn = dn0), orbitals = o) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) + hopping(t; dn = dn0, forcehermitian = false), orbitals = o) isa Hamiltonian
        end
    end
end

@testset "orbitals and sublats" begin
    orbs = (:a, (:a,), (:a, :b, 3), ((:a, :b), :c), ((:a, :b), (:c,)), (Val(2), Val(1)),
            (:A => (:a, :b), :D => :c), :D => Val(4))
    lat = LatticePresets.honeycomb()
    for orb in orbs
        @test hamiltonian(lat, onsite(I), orbitals = orb) isa Hamiltonian
    end
    @test hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = (:A,:B)),
                      orbitals = :B => Val(2)) isa Hamiltonian
    h1 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = (:A,:B)),
                      orbitals = :B => Val(2))
    h2 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = ((:A,:B),)),
                      orbitals = :B => Val(2))
    @test bloch(h1, 1, 2) == bloch(h2, 1, 2)
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

@testset "modifiers" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(0)) |> unitcell(2, modifiers = onsite!((o, r) -> 1))
    @test diag(bloch(h)) == ComplexF64[1, 1, 1, 1, 1, 1, 1, 1]
end