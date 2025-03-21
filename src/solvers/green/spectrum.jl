############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct InverseGreen0DEigenSolver{I<:InverseGreen0D, S<:GS.Spectrum}
    invgreen::I
    solver::S
end

struct AppliedSpectrumGreenSolver{S<:Union{Eigen,InverseGreen0DEigenSolver}} <: AppliedGreenSolver
    eigen::S
end

struct SpectrumGreenSlicer{C,E<:Eigen} <: GreenSlicer{C}
    ω::C
    bare_eigen::E
end

#region ## Constructor ##


#end

#region ## API ##

minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:InverseGreen0DEigenSolver}, parentham, parentcontacts) =
    apply(s.eigen.solver, parentham, parentcontacts)
# parentham needs to be non-parametric and s.eigen is its eigenspectrum, so no need to recompute
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver{<:Eigen}, parentham, parentcontacts) = s

## apply ##

apply(s::GS.Spectrum, h::Hamiltonian{<:Any,<:Any,0}, ::EmptyContacts) =
    AppliedSpectrumGreenSolver(eigen(spectrum(h; solver = s.solver)))

apply(s::GS.Spectrum, h::AbstractHamiltonian0D, c::Contacts) =
    AppliedSpectrumGreenSolver(InverseGreen0DEigenSolver(inverse_green(h, c), s))

apply(::GS.Spectrum, h::AbstractHamiltonian, ::Contacts) =
    argerror("Can only use GS.Spectrum with bounded (L=0) AbstractHamiltonians. Received one with lattice dimension L=$(latdim(h))")

## bare_eigen and dressed_eigen ##

# get Eigen for H0D (without contacts)
bare_eigen(s::AppliedSpectrumGreenSolver{<:Eigen}; params...) = s.eigen

function bare_eigen(s::AppliedSpectrumGreenSolver{<:InverseGreen0DEigenSolver}; params...)
    h = hamiltonian(s.eigen.invgreen)
    mat = call!(h, SA[]; params...)
    return get_eigen(s.eigen.solver.solver, mat, h)
end

# get Eigen for H0D + Σ(ω)  (with contacts but for fixed ω)
dressed_eigen(s::AppliedSpectrumGreenSolver{<:Eigen}, ω; params...) = s.eigen

function dressed_eigen(s::AppliedSpectrumGreenSolver{<:InverseGreen0DEigenSolver}, ω; params...)
    g⁻¹ = s.eigen.invgreen
    mat = call!(g⁻¹, ω; params...)
    eigen = get_eigen(s.eigen.solver.solver, mat, g⁻¹)
    # convert from ω-ϵᵢ to ϵᵢ
    ϵs, _ = eigen
    @. ϵs = ω - ϵs
    return eigen
end

function get_eigen(solver, mat, h)
    mat´ = ES.input_matrix(solver, h)
    mat´ === mat || copy!(mat´, mat)
    # mat´ could be dense, while mat is sparse, so if not egal, we copy
    # the solver always receives the type of matrix mat´ declared by ES.input_matrix
    eigen = solver(mat´)
    # _, psis = eigen
    # orthonormalize!(psis)
    return eigen
end

## call ##

# contacts are incorporated at each ω using a T-matrix
function build_slicer(s::AppliedSpectrumGreenSolver, g, ω, Σblocks, corbitals; params...)
    eigen = bare_eigen(s; params...)
    g0slicer = SpectrumGreenSlicer(complex(ω), eigen)
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
    es, psis = s.bare_eigen
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
    orbaxes = orbrows(gs), orbcols(gs)
    solver = DensityMatrixSpectrumSolver(gs, orbaxes, s)
    return DensityMatrix(solver, gs)
end

## call

# We get the eigenpairs of H0D + Σ(ω = µ), and sum with populations
function (s::DensityMatrixSpectrumSolver)(µ, kBT; params...)
    bs = blockstructure(s.gs)
    es, psis = dressed_eigen(s.solver, µ; params...)
    β = inv(kBT)
    fs = (@. fermi(es-µ, β))
    fpsis = psis .* transpose(fs)
    ρcell = EigenProduct(bs, psis, fpsis)
    result = call!_output(s.gs)
    getindex!(result, ρcell, s.orbaxes...)
    return result
end

#endregion
#endregion
