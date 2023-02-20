############################################################################################
# Green solvers
#   All new S::AbstractGreenSolver must live in the GreenSolvers module, and must implement
#     - apply(s, h::AbstractHamiltonian, c::Contacts) -> AppliedGreenSolver
#   All new s::AppliedGreenSolver must implement
#      - s(ω, Σblocks, ::ContactBlockStructure) -> AbstractGreenSlicer
#      - Optional: minimal_callsafe_copy(gs) -> has a deepcopy fallback
#   This GreenSolution provides in particular:
#      - GreenSlicer to compute e.g. G[gi, gi´]::AbstractMatrix for indices gi, see below
#      - linewidth flat matrix Γᵢ for each contact
#      - LatticeSlice for merged contacts
#      - bs::ContactBlockStructure for contacts LatticeSlice
#   All gs::GreenSlicer's must implement
#      - view(gs, ::ContactIndex, ::ContactIndex) -> g(ω; kw...) between specific contacts
#      - view(gs, ::Colon, ::Colon) -> g(ω; kw...) between all contacts
#      - gs[i::CellOrbitals, j::CellOrbitals] -> must return a Matrix for type stability
#      - Optional: minimal_callsafe_copy(gs) -> has a deepcopy fallback
#   The user-facing indexing API accepts:
#      - contact(i)::ContactIndex -> Sites of Contact number i
#      - cellsites(cell::Tuple, sind::Int)::Subcell -> Single site in a cell
#      - cellsites(cell::Tuple, sindcollection)::Subcell -> Site collection in a cell
#      - cellsites(cell::Tuple, slat::Symbol)::Subcell -> Whole sublattice in a cell
#      - cellsites(cell::Tuple, :) ~ cell::Union{NTuple,SVector} -> All sites in a cell
#      - sel::SiteSelector ~ NamedTuple -> forms a LatticeSlice
############################################################################################

############################################################################################
# SelfEnergy solvers
#   All s::AbstractSelfEnergySolver must support the call! API
#     - call!(s::RegularSelfEnergySolver, ω; params...) -> Σreg::AbstractMatrix
#     - call!(s::ExtendedSelfEnergySolver, ω; params...) -> (Vᵣₑ, gₑₑ⁻¹, Vₑᵣ) AbsMats
#         With the extended case, the equivalent Σreg reads Σreg = VᵣₑgₑₑVₑᵣ
#     - call!_output(s::AbstractSelfEnergySolver) -> object returned by call!(s, ω; kw...)
#     - minimal_callsafe_copy(s::AbstractSelfEnergySolver) ->  optional, deepcopy fallback
#   These AbstractMatrices are flat, defined on the LatticeSlice in parent SelfEnergy
#       Note: `params` are only needed in cases where s adds new parameters that must be
#       applied (e.g. SelfEnergyModelSolver). Otherwise one must assume that any parent
#       ParametricHamiltonian to GreenFunction has already been call!-ed before calling s.
############################################################################################

############################################################################################
# SelfEnergy constructors
#   For each attach(h, sargs...; kw...) syntax we need, we must implement:
#     - SelfEnergy(h::AbstractHamiltonian, sargs...; kw...) -> SelfEnergy
#   SelfEnergy wraps the corresponding SelfEnergySolver, be it Regular or Extended
############################################################################################

module GreenSolvers

using Quantica: AbstractGreenSolver

struct SparseLU <:AbstractGreenSolver end

struct Schur{T<:AbstractFloat} <: AbstractGreenSolver
    shift::T                      # Tunable parameter in algorithm, see Ω in scattering.pdf
    boundary::T                   # Cell index for boundary (float to allow boundary at Inf)
end

Schur(; shift = 1.0, boundary = Inf) = Schur(shift, float(boundary))

end # module

const GS = GreenSolvers

include("greensolvers/selfenergymodel.jl")
include("greensolvers/sparselu.jl")
include("greensolvers/schur.jl")
# include("greensolvers/bands.jl")

