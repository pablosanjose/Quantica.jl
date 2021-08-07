precompile(LatticePresets.linear, ())
precompile(LatticePresets.square, ())
precompile(LatticePresets.triangular, ())
precompile(LatticePresets.honeycomb, ())
precompile(LatticePresets.cubic, ())
precompile(LatticePresets.fcc, ())
precompile(LatticePresets.bcc, ())
precompile(LatticePresets.hcp, ())

for E in 0:3, L in 0:E, L´ in 0:L
    precompile(supercell, (Lattice{Float64,E,L}, Vararg{NTuple{L,Int},L´}))
end
