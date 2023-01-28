############################################################################################
# GreenSolvers module
#   All new S::AbstractGreenSolver must implement
#     - apply(s, h::OpenHamiltonian, c::Contacts) -> AppliedGreenSolver
#   All new s::AppliedGreenSolver must implement
#      - s(ω, Σblocks, orbital_blockstruct) -> AbstractGreenSlicer
#      - minimal_callsafe_copy(gs) -> has a deepcopy fallback
#   This GreenSolution provides in particular:
#      - GreenSlicer to compute e.g. G[gi, gi´]::AbstractMatrix for indices gi, see below
#      - linewidth flat matrix Γᵢ for each contact
#      - LatticeSlice for merged contacts
#      - bscontacts::ContactBlockStructure for contacts LatticeSlice
#   All gs::GreenSlicer's must implement
#      - view(gs, ::ContactIndex, ::ContactIndex) -> g(ω; kw...) between specific contacts
#      - view(gs, ::Colon, ::Colon) -> g(ω; kw...) between all contacts
#      - gs[i::CellOrbitals, j::CellOrbitals]
#      - minimal_callsafe_copy(gs) -> has a deepcopy fallback
#   The user-facing indexing API accepts:
#      - contact(i)::ContactIndex -> Sites of Contact number i
#      - cellsites(cell::Tuple, sind::Int)::Subcell -> Single site in a cell
#      - cellsites(cell::Tuple, sindcollection)::Subcell -> Site collection in a cell
#      - cellsites(cell::Tuple, slat::Symbol)::Subcell -> Whole sublattice in a cell
#      - cellsites(cell::Tuple, :) ~ cell::Union{NTuple,SVector} -> All sites in a cell
#      - sel::SiteSelector ~ NamedTuple -> forms a LatticeSlice
#region

module GreenSolvers

using Quantica: AbstractGreenSolver

struct SparseLU <:AbstractGreenSolver end

end # module

const GS = GreenSolvers

include("greensolvers/sparselu.jl")
# include("greensolvers/schur.jl")
# include("greensolvers/bands.jl")

