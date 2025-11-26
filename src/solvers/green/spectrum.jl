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
# SpectralSum
#   a struct for generic spectral sums in 0D systems. Mostly for use in observables.
#   If ss = SpectralSum(gs::GreenSlice; inplace = true), then S === gs.output is
#     S = ⟨is| \sum_n kernel(dist(real(ε_n)), |ψ_n⟩) |js⟩ = ss(kernel!, dist; params...),
#   where ε_n, |ψ_n⟩ are eigenpairs of the parent Hamiltonian and `is`, `js` are orbslices.
#   If `inplace = true` we must provide a `kernel!(partialsum, f_n, |ψ_n⟩)` function that
#   adds `kernel(f_n, |ψ_n⟩)` in-place to `partialsum`.
#   If kernel is missing, it defaults to `kernel(f_n, |ψ_n⟩) = f_n * |ψ_n⟩⟨ψ_n|` although
#   the implementation uses an `EigenProduct` optimization in this case.
#region

struct SpectralSum{G<:GreenSlice,M}
    gs::G
    inplace::Bool
    temp::M
end

function SpectralSum(gs::GreenSlice; inplace = true)
    has_selfenergy(gs) && argerror("SpectrumSum currently support only `nothing` contacts")
    solver(parent(gs)) isa AppliedSpectrumGreenSolver || argerror("SpectralSum only supports AppliedSpectrumGreenSolver")
    if inplace
        temp = similar(call!_output(gs), size(parent(gs)))
        return SpectralSum(gs, inplace, temp)
    else
        return SpectralSum(gs, inplace, missing)
    end
end

(ss::SpectralSum)(µ::Real, kBT::Real; params...) =
    ss(missing, ε -> fermi(ε - µ, inv(kBT)); params...)

(ss::SpectralSum)(kernel!, dist; params...) =
    ss(kernel!, dist, spectrum(solver(parent(ss.gs)); params...))

# missing kernel!: kernel!(temp, f_n, |ψ_n⟩) = f_n * |ψ_n⟩⟨ψ_n|, EigenProduct optimization
function (ss::SpectralSum)(::Missing, dist, (εs, psis))
    gs = ss.gs
    fs = dist.(real.(εs))
    fpsis = psis .* transpose(fs)
    bs = blockstructure(gs)
    sp_sum = EigenProduct(bs, psis, fpsis)
    return project_sum(gs, sp_sum)
end

# general kernel!
function (ss::SpectralSum)(kernel!, dist, (εs, psis))
    gs = ss.gs
    T = eltype(psis)
    if ss.inplace
        sp_sum = fill!(ss.temp, zero(T))
        for (ε, φ) in zip(εs, eachcol(psis))
            kernel!(sp_sum, dist(real(ε)), φ)
        end
    else
        sp_sum = sum(kernel!(dist(real(ε)), φ) for (ε, φ) in zip(εs, eachcol(psis)))
    end
    return project_sum(gs, sp_sum)
end

function project_sum(gs, sp_sum)
    result = call!_output(gs)
    orows, ocols = orbrows(gs), orbcols(gs)
    getindex!(result, sp_sum, orows, ocols)
    return result
end

############################################################################################
# densitymatrix
#   specialized densitymatrix method for GS.Spectrum in terms of SpectralSum
#region

struct DensityMatrixSpectrumSolver{T,G<:GreenSlice{T}}
    ssum::SpectralSum{G}
end

## Constructor

function densitymatrix(::AppliedSpectrumGreenSolver, gs::GreenSlice)
    solver = DensityMatrixSpectrumSolver(SpectralSum(gs))
    return DensityMatrix(solver, gs)
end

## call

(s::DensityMatrixSpectrumSolver)(µ, kBT; params...) = s.ssum(µ, kBT; params...)

#endregion
#endregion
