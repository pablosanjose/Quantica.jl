############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct AppliedSpectrumGreenSolver{B,S<:Spectrum{<:Any,B}} <:AppliedGreenSolver
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

minimal_callsafe_copy(s::AppliedSpectrumGreenSolver) = AppliedSpectrumGreenSolver(s.spectrum)

minimal_callsafe_copy(s::SpectrumGreenSlicer) = SpectrumGreenSlicer(s.ω, s.solver)

#endregion

#region ## apply ##

apply(s::GS.Spectrum, h::AbstractHamiltonian0D, ::Contacts) =
    AppliedSpectrumGreenSolver(spectrum(h; s.spectrumkw...))

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

#endregion
