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
    precompile(nrange, (Int, Lattice{T,E,L}))
    precompile(site, (Lattice{T,E,L}, Int, SVector{L,Int}))
end

# for N in 1:2
#     precompile(iterate, (BoxIterator{N}, BoxIteratorState{N}))
# end
