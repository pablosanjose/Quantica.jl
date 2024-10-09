############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct AppliedSpectrumGreenSolver{B,S<:Spectrum{<:Any,B}} <: AppliedGreenSolver
    spectrum::S
end

struct SpectrumGreenSlicer{C,B,S<:AppliedSpectrumGreenSolver{B}} <: GreenSlicer{C}
    ω::C
    solver::S
end

#region ## Constructor ##


#end

#region ## API ##

spectrum(s::AppliedSpectrumGreenSolver) = s.spectrum

# Parent ham needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(s.spectrum)

minimal_callsafe_copy(s::SpectrumGreenSlicer, parentham, parentcontacts) =
    SpectrumGreenSlicer(s.ω, s.solver)

#endregion

#region ## apply ##

apply(s::GS.Spectrum, h::Hamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(spectrum(h; s.spectrumkw...))

apply(s::GS.Spectrum, h::ParametricHamiltonian, ::Contacts) =
    argerror("Cannot use GS.Spectrum with ParametricHamiltonian. Apply parameters with h(;params...) first.")

apply(::GS.Spectrum, h::AbstractHamiltonian, ::Contacts) =
    argerror("Can only use Spectrum with bounded Hamiltonians")

#endregion

#region ## call ##

function (s::AppliedSpectrumGreenSolver)(ω, Σblocks, corbitals)
    g0slicer = SpectrumGreenSlicer(complex(ω), s)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

#endregion

#endregion

############################################################################################
# SpectrumGreenSlicer indexing
#   We don't implement view over contacts because that is taken care of by TMatrixSlicer
#region

function Base.getindex(s::SpectrumGreenSlicer{C}, i::CellOrbitals{0}, j::CellOrbitals{0}) where {C}
    oi, oj = orbindices(i), orbindices(j)
    es, psis = spectrum(s.solver)
    vi, vj = view(psis, oi, :), view(psis, oj, :)
    vj´ = vj' ./ (s.ω .- es)
    return vi * vj´
end

#endregion

############################################################################################
# densitymatrix
#   specialized DensityMatrix method for GS.Spectrum
#region

struct DensityMatrixSpectrumSolver{T,A,G<:GreenFunction,P,R}
    g::G                        # parent of GreenSlice
    es::Vector{Complex{T}}
    axes::A
    psis::P
    fpsis::P
    ρmat::R
end

## Constructor

function densitymatrix(s::AppliedSpectrumGreenSolver, gs::GreenSlice)
    # SpectrumGreenSlicer is 0D, so there is a single cellorbs in dict.
    # If rows/cols are contacts, we need their orbrows/orbcols (unlike for gs(ω; params...))
    i, j = onlycellorbs(orbrows(gs)), onlycellorbs(orbcols(gs))
    es, psis = spectrum(s)
    fpsis = copy(psis')
    ρmat = similar_Array(gs)
    g = parent(gs)
    solver = DensityMatrixSpectrumSolver(g, es, (i,j), psis, fpsis, ρmat)
    return DensityMatrix(solver, gs)
end

onlycellorbs(orb::AnyCellOrbitals) = orb
onlycellorbs(orb::AnyOrbitalSlice) = only(cellsdict(orb))
onlycellorbs(orb::DiagIndices) = onlycellorbs(parent(orb))

## call

function (s::DensityMatrixSpectrumSolver)(mu, kBT; params...)
    psis, fpsis, es = s.psis, s.fpsis, s.es
    β = inv(kBT)
    @. fpsis = fermi(es - mu, β) .* psis'
    ρmat = fill_rho_blocks!(s, psis, fpsis, s.axes...)
    return copy(ρmat)
end

function fill_rho_blocks!(s::DensityMatrixSpectrumSolver, psis, fpsis, i, j)
    vpsis = view(psis, orbindices(i), :)
    vfpsis = view(fpsis, :, orbindices(j))
    return mul!(s.ρmat, vpsis, vfpsis)
end

# this relies on the apply_kernel method from the schur.jl densitymatrix solver
fill_rho_blocks!(s::DensityMatrixSpectrumSolver, psis, fpsis, di::DiagIndices, ::DiagIndices) =
    append_diagonal!(empty!(s.ρmat), FermiEigenstates(psis, fpsis), parent(di), kernel(di), s.g)

#endregion
#endregion
