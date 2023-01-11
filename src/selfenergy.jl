############################################################################################
# IJVBuilder!
#   Flat in-place sparse matrix builder, similar to IJVBuilder, but with MultiBlockStructure
#region

struct IJVBuilder!{C,L}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{C}
    m::Int
    n::Int
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{C}
    klasttouch::Vector{Int}
    blockstruct::MultiBlockStructure{L}
end

#region ## Constructor ##

function IJVBuilder!{C}(bs::MultiBlockStructure) where {C}
    m = n = flatsize(bs)
    I, J, V = Int[], Int[], C[]
    csrrowptr, csrcolval, csrnzval = Vector{Int}(undef, m + 1), Int[], C[]
    klasttouch = Vector{Int}(undef, n)
    return IJVBuilder!(I, J, V, m, n, csrrowptr, csrcolval, csrnzval, klasttouch, bs)
end

#endregion

#region ## API ##

function sparse!(b::IJVBuilder!)
    coolen = length(b.I)
    resize!(b.csrcolval, coolen)
    resize!(b.csrnzval, coolen)
    s = SparseArrays.sparse!(b.I, b.J, b.V, b.m, b.n, +,
        b.klasttouch, b.csrrowptr, b.csrcolval, b.csrnzval,
        b.I, b.J, b.V)
    return s
end

function Base.empty!(b::IJVBuilder!)
    empty!(b.I)
    empty!(b.J)
    empty!(b.V)
    return b
end

function Base.setindex!(b::IJVBuilder!, vs, iunflat::Integer, junflat::Integer)
    bs = b.blockstruct
    is = siterange(bs, iunflat)
    js = siterange(bs, junflat)
    leni, lenj = length(is), length(js)
    checkblocksize(vs, (leni, lenj))  # tools.jl
    for c in CartesianIndices((is, js))
        i, j = Tuple(c)
        push!(b.I, i)
        push!(b.J, j)
        push!(b.V, vs[i, j])
    end
    return b
end


# _getindex(vs::UniformScaling, i, j) = ifelse(i == j, vs.λ, zero(vs.λ))
# _getindex(vs, i, j) = vs[i, j]

#endregion
#endregion
#endregion top

############################################################################################
# SelfEnergyModel <: RegularSelfEnergySolver <: AbstractSelfEnergySolver
#region

struct SelfEnergyModel{T,E,L,A<:AppliedParametricModel} <: RegularSelfEnergySolver
    model::A
    latslice::LatticeSlice{T,E,L}
    kdtree::KDTree{SVector{E,T},Euclidean,T}  # needed to efficiently apply hopping terms
    builder::IJVBuilder!{Complex{T},L}
end

#region ## API ##

SelfEnergy(h::AbstractHamiltonian, model::ParametricModel; kw...) =
    SelfEnergy(h, model, siteselector(; kw...))

function SelfEnergy(h::AbstractHamiltonian{T}, model::ParametricModel, sel::SiteSelector) where {T}
    lat = lattice(h)
    asel = apply(sel, lat)
    latslice = lat[asel]
    amodel = apply(model, lat)
    bs = MultiBlockStructure(latslice, h)
    builder = IJVBuilder!{Complex{T}}(bs)
    kdtree = KDTree(collect(sites(latslice)))
    solver = SelfEnergyModel(amodel, latslice, kdtree, builder)
    return SelfEnergy(solver, latslice)
end

function call!(s::SelfEnergyModel{T}, ω; params...) where {T}
    empty!(s.builder)
    foreach(terms(s.model)) do term
        apply_term!(s, term, ω; params...)
    end
    return sparse!(s.builder)
end

function apply_term!(s::SelfEnergyModel, o::ParametricOnsiteTerm, ω; params...)
    foreach_site(selector(o), s.latslice) do i, r, n, islice
        s.builder[islice, islice] = o(ω, r; params...)
    end
    return s
end

function apply_term!(s, t::ParametricHoppingTerm, ω; params...)
    foreach_hop(selector(t), s.latslice, s.kdtree) do is, (r, dr), ns, (islice, jslice)
        s.builder[islice, jslice] = t(ω, r, dr; params...)
    end
    return s
end

#endregion
#endregion

