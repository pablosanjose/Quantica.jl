############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct SpectrumSolver{H<:ParametricHamiltonian{<:Any,<:Any,0}, S<:GS.Spectrum}
    h::H
    solver::S
end

struct AppliedSpectrumGreenSolver{S<:Union{Spectrum,SpectrumSolver}} <: AppliedGreenSolver
    spectrum::S
end

struct SpectrumGreenSlicer{C,S<:Spectrum} <: GreenSlicer{C}
    ω::C
    spectrum::S
end

#region ## Constructor ##


#end

#region ## API ##

# Parent ham needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:Spectrum}, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(s.spectrum)
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(SpectrumSolver(parentham, s.spectrum.solver))

#endregion

#region ## apply ##

apply(s::GS.Spectrum, h::Hamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(spectrum(h; s.spectrumkw...))

apply(s::GS.Spectrum, h::ParametricHamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(SpectrumSolver(h, s))

apply(::GS.Spectrum, h::AbstractHamiltonian, ::Contacts) =
    argerror("Can only use GS.Spectrum with bounded (L=0) AbstractHamiltonians. Received one with lattice dimension L=$(latdim(h))")

#endregion

#region ## call ##

function build_slicer(s::AppliedSpectrumGreenSolver, g, ω, Σblocks, corbitals; params...)
    sp = spectrum(s; params...)
    g0slicer = SpectrumGreenSlicer(complex(ω), sp)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

# get spectrum from solver
spectrum(s::AppliedSpectrumGreenSolver{<:Spectrum}; params...) =
    s.spectrum
spectrum(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}; params...) =
    spectrum(call!(s.spectrum.h; params...); s.spectrum.solver.spectrumkw...)

# FixedParamGreenSolver support
maybe_apply_params(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}, h, c; params...) =
    AppliedSpectrumGreenSolver(spectrum(call!(h; params...); s.spectrum.solver.spectrumkw...))

#endregion

#endregion

############################################################################################
# SpectrumGreenSlicer indexing
#   We don't implement view over contacts because that is taken care of by TMatrixSlicer
#region

function Base.getindex(s::SpectrumGreenSlicer{C}, i::CellOrbitals{0}, j::CellOrbitals{0}) where {C}
    oi, oj = orbindices(i), orbindices(j)
    es, psis = s.spectrum
    vi, vj = view(psis, oi, :), view(psis, oj, :)
    vj´ = vj' ./ (s.ω .- es)
    return vi * vj´
end

#endregion

############################################################################################
# densitymatrix
#   specialized DensityMatrix method for GS.Spectrum
#region

struct DensityMatrixSpectrumSolver{T,G<:GreenSlice{T},A,S<:AppliedSpectrumGreenSolver}
    gs::G
    orbaxes::A
    solver::S
end

## Constructor

function densitymatrix(s::AppliedSpectrumGreenSolver, gs::GreenSlice)
    # SpectrumGreenSlicer is 0D, so there is a single cellorbs in dict.
    # If rows/cols are contacts, we need their orbrows/orbcols (unlike for gs(ω; params...))
    has_selfenergy(gs) && argerror("The Spectrum densitymatrix solver currently support only `nothing` contacts")
    orbaxes = orbrows(gs), orbcols(gs)
    solver = DensityMatrixSpectrumSolver(gs, orbaxes, s)
    return DensityMatrix(solver, gs)
end

## call

function (s::DensityMatrixSpectrumSolver)(µ, kBT; params...)
    bs = blockstructure(s.gs)
    es, psis = spectrum(s.solver; params...)
    β = inv(kBT)
    fs = (@. fermi(es - µ, β))
    fpsis = psis .* transpose(fs)
    ρcell = EigenProduct(bs, psis, fpsis)
    result = call!_output(s.gs)
    getindex!(result, ρcell, s.orbaxes...)
    return result
end

#endregion
#endregion
