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
    # precompile(nrange, (Int, Lattice{T,E,L}))
    # precompile(site, (Lattice{T,E,L}, Int, SVector{L,Int}))
    # O = Complex{T}
    # Ms = (
    #     Quantica.HoppingTerm{Int64, Quantica.HopSelector{Missing, Missing, Missing, Missing, Quantica.NeighborRange}, Int64},
    #     Quantica.HoppingTerm{Float64, Quantica.HopSelector{Missing, Missing, Missing, Missing, Quantica.NeighborRange}, Int64},
    #     Quantica.OnsiteTerm{Int64, Quantica.SiteSelector{Missing, Missing, Missing}, Int64},
    #     Quantica.OnsiteTerm{Float64, Quantica.SiteSelector{Missing, Missing, Missing}, Int64}
    # )
    # for M in Ms
    #     precompile(applyterm!, (IJVBuilder{T,E,L,O}, M))
    # end
end

# for N in 1:2
#     precompile(iterate, (BoxIterator{N}, BoxIteratorState{N}))
# end
