############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct SpectrumSolver{H<:ParametricHamiltonian{<:Any,<:Any,0}, S<:GS.Spectrum}
    h::H
    solver::S
end

struct AppliedSpectrumGreenSolver{S<:Union{Spectrum,SpectrumSolver}} <: AppliedGreenSolver
    spectrum::S
    ishermitian::Bool   # false if S<:SpectrumSolver, since we still don't know if ishermitian(h(; params...))
end

struct SpectrumGreenSlicer{C,S<:Spectrum} <: GreenSlicer{C}
    ω::C
    spectrum::S
    ishermitian::Bool
end

#region ## API ##

# parentham needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:Spectrum}, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(s.spectrum, ishermitian(parentham))
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(SpectrumSolver(parentham, s.spectrum.solver), false)

LinearAlgebra.ishermitian(s::AppliedSpectrumGreenSolver{<:Spectrum}; _...) = s.ishermitian
LinearAlgebra.ishermitian(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}; params...) =
    ishermitian(call!(s.spectrum.h; params...))
LinearAlgebra.ishermitian(s::SpectrumGreenSlicer) = s.ishermitian

#endregion

#region ## apply ##

apply(s::GS.Spectrum, h::Hamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(spectrum(h; s.spectrumkw...), ishermitian(h))

apply(s::GS.Spectrum, h::ParametricHamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(SpectrumSolver(h, s), false)

apply(::GS.Spectrum, h::AbstractHamiltonian, ::Contacts) =
    argerror("Can only use GS.Spectrum with bounded (L=0) AbstractHamiltonians. Received one with lattice dimension L=$(latdim(h))")

#endregion

#region ## call ##

function build_slicer(s::AppliedSpectrumGreenSolver, g, ω, Σblocks, corbitals; params...)
    sp = spectrum(s; params...)
    g0slicer = SpectrumGreenSlicer(complex(ω), sp, ishermitian(s; params...))
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

# get spectrum from solver
spectrum(s::AppliedSpectrumGreenSolver{<:Spectrum}; params...) =
    s.spectrum
spectrum(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}; params...) =
    spectrum(call!(s.spectrum.h; params...); s.spectrum.solver.spectrumkw...)

# FixedParamGreenSolver support
function maybe_apply_params(s::AppliedSpectrumGreenSolver{<:SpectrumSolver}, h, c; params...)
    h´ = call!(h; params...)
    sp = spectrum(h´; s.spectrum.solver.spectrumkw...)
    return AppliedSpectrumGreenSolver(sp, ishermitian(h´))
end

#endregion

#endregion

############################################################################################
# SpectrumGreenSlicer indexing
#   We don't implement view over contacts because that is taken care of by TMatrixSlicer
#region

function Base.getindex(s::SpectrumGreenSlicer, i::CellOrbitals{0}, j::CellOrbitals{0})
    oi, oj = orbindices(i), orbindices(j)
    es, psis = s.spectrum
    ωes = (s.ω .- es)
    vi = view(psis, oi, :) ./ transpose(ωes)
    sij = ishermitian(s) ? vi * view(psis, oj, :)' : vi * lazypseudoinverse(view(psis, oj, :))
    return sij
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
    result = call!_output(s.gs)
    if ishermitian(s.solver)
        getindex!(result, EigenProduct(bs, fpsis, psis'), s.orbaxes...)
    else
        getindex!(result, EigenProduct(bs, fpsis, lazypseudoinverse(psis)), s.orbaxes...)
    end
    return result
end

#endregion
#endregion
