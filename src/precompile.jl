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

    Bs = (Complex{T}, Base.tail(ntuple(N -> SMatrix{N,N,Complex{T},N*N}, Val(4)))...)
    for (N, B) in enumerate(Bs)
        for L´ in 0:L
            precompile(supercell_harmonics, (Hamiltonian{T,E,L,B}, SupercellData{T,E,L,L´,L*L´}, CSCBuilder{T,E,L´,B}, Int))
        end
        precompile(OrbitalBlockStructure{B}, (Val{N}, Vector{Int}))
        precompile(applyterm!, (IJVBuilder{T,E,L,B}, AppliedOnsiteTerm{T,E,L,B}))
        precompile(applyterm!, (IJVBuilder{T,E,L,B}, AppliedHoppingTerm{T,E,L,B}))
    end
end

for L in 1:2
    T = Float64
    B = ComplexF64
    precompile(bands_precompilable, (Vector{EigenSolver{T,L,B}}, Mesh{SVector{L,T}}, Bool, Vector{SVector{L,T}}, Int, T, Bool, Bool))
end