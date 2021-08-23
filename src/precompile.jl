precompile(LatticePresets.linear, ())
precompile(LatticePresets.square, ())
precompile(LatticePresets.triangular, ())
precompile(LatticePresets.honeycomb, ())
precompile(LatticePresets.cubic, ())
precompile(LatticePresets.fcc, ())
precompile(LatticePresets.bcc, ())
precompile(LatticePresets.hcp, ())

for E in 0:2, L in 0:E
    T = Float64
    for L´ in 0:L
        precompile(supercell, (Lattice{T,E,L}, Vararg{NTuple{L,Int},L´}))
    end
    precompile(supercell, (Lattice{T,E,L}, Int))
    precompile(nrange, (Int, Lattice{T,E,L}))
    precompile(site, (Lattice{T,E,L}, Int, SVector{L,Int}))
    O = Complex{T}
    precompile(supercell, (Hamiltonian{T,E,L,O}, Int))
    precompile(applyterm!, (IJVBuilder{T,E,L,O},AppliedOnsiteTerm{T,E,O}))
    precompile(applyterm!, (IJVBuilder{T,E,L,O},AppliedHoppingTerm{T,E,L,O}))
    # precompile(Hamiltonian{T,E,L,O}, (Lattice{T,E,L},OrbitalStructure{O},Vector{HamiltonianHarmonic{L,O}}))
    for N = 2:4
        O = SMatrix{N,N,Complex{T},N*N}
        for L´ in 0:L
            precompile(supercell, (Hamiltonian{T,E,L,O}, Vararg{NTuple{L,Int},L´}))
        end
        precompile(applyterm!, (IJVBuilder{T,E,L,O},AppliedOnsiteTerm{T,E,O}))
        precompile(applyterm!, (IJVBuilder{T,E,L,O},AppliedHoppingTerm{T,E,L,O}))
        # precompile(Hamiltonian{T,E,L,O}, (Lattice{T,E,L},OrbitalStructure{O},Vector{HamiltonianHarmonic{L,O}}))
    end
end

# for N in 1:2
#     precompile(iterate, (BoxIterator{N}, BoxIteratorState{N}))
# end
