using Quantica, ArnoldiMethod, LinearAlgebra, QuadGK, NumericalIntegration, FFTW

h = hamiltonian(unitcell(LatticePresets.linear(a0 = .001), 
    region = r -> abs(r[1])<= 1), onsite(r -> r[1]))
#testing KPM
l = dosKPM(h, order = 20000)

plotKPMdos(l) = plotKPMdos(l[1], l[2])
function plotKPMdos(x, y)
    f = Figure(resolution = (300,300))
    ax = Axis(f[1,1])
    lines!(ax, x, y)
    xlims!(ax, (-0.02,0.02))
    ylims!(ax,(0,6))
    f
end

h = hamiltonian(unitcell(LatticePresets.linear(a0 = .001), 
    region = r -> abs(r[1])<= 1), onsite(r -> @SMatrix[r[1] 0; 0 r[1]]), orbitals = Val(2))

sp = spectrum(h);
# hist(sp.energies[abs.(sp.energies) .< 0.02], bins = 1000 ) #exact


##########
# Degenerate eigenvalues
# 

#
#
##########