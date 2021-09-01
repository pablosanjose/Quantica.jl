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
        S = SMatrix{L,L´,Int,L*L´}
        precompile(supercell_data, (Lattice{T,E,L}, S, SVector{L,Int}, AppliedSiteSelector{T,E,L}))
    end
    precompile(nrange, (Int, Lattice{T,E,L}))
    precompile(site, (Lattice{T,E,L}, Int, SVector{L,Int}))

    Os = (Complex{T}, Base.tail(ntuple(N -> SMatrix{N,N,Complex{T},N*N}, Val(4)))...)
    for (N, O) in enumerate(Os)
        for L´ in 0:L
            precompile(supercell_harmonics, (Hamiltonian{T,E,L,O}, SupercellData{T,E,L,L´,L*L´}, CSCBuilder{T,E,L´,O}, Int))
        end
        precompile(OrbitalStructure{O}, (Lattice{T,E,L}, Val{N}))
        precompile(applyterm!, (IJVBuilder{T,E,L,O}, AppliedOnsiteTerm{T,E,L,O}))
        precompile(applyterm!, (IJVBuilder{T,E,L,O}, AppliedHoppingTerm{T,E,L,O}))
    end
end