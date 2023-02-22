############################################################################################
# Coupler -  must be defined after IJVBuilder
#region

struct Coupler{T,E,L,B,N,HS<:NTuple{N,AbstractHamiltonian}} <: AbstractMatrix{Symbol}
    hs::HS                                  # must be NTuple because eltype not concrete
    lattice::Lattice{T,E,L}
    builder::IJVBuilder{T,E,L,B}
    blockoffsets::Tuple{Int,Vararg{Int,N}}  # must be NTuple to broadcast statically with hs
    assigned::Matrix{Bool}
end

#region ## API ##

Base.getindex(c::Coupler, i...) = c.assigned[i...] ? :assigned : :unassigned

function Base.setindex!(c::Coupler, value, i, j)
    if i != j && i in axes(c,1) && j in axes(c, 2)
        coupleblock!(c, value, i, j)
        c.assigned[i, j] = true
    else
        throw(ArgumentError("Can only couple existing off-diagonal blocks"))
    end
    return value
end

Base.BroadcastStyle(::Type{Coupler}) = Broadcast.ArrayStyle{Coupler}()

Base.axes(c::Coupler, i...) = axes(c.assigned, i...)
Base.size(c::Coupler, i...) = size(c.assigned, i...)

lattice(c::Coupler) = c.lattice

blocks(c::Coupler) = c.hs

block(c::Coupler, i) = c.hs[i]

builder(c::Coupler) = c.builder

blockoffsets(c::Coupler) = c.blockoffsets

blockranges(c::Coupler, i, j) =
    c.blockoffsets[i]+1:c.blockoffsets[i+1], c.blockoffsets[j]+1:c.blockoffsets[j+1]

assignedblocks(c::Coupler) = c.assigned

blockstructure(c::Coupler) = blockstructure(builder(c))

#endregion

############################################################################################
# coupler
#region

function coupler(hs::AbstractHamiltonian...; all = missing)
    c = coupler_unflat(hs)
    if all !== missing
        for j in axes(c, 2), i in axes(c, 1)
            i == j && continue
            c[i, j] = all
        end
    end
    return c
end

coupler_unflat(hs::Tuple) = coupler_unflat(promote_type(typeof.(hs)...), hs)

function coupler_unflat(::Type{<:AbstractHamiltonian{T,E,L,B}}, hs::NTuple{N,AbstractHamiltonian}) where {T,E,L,B,N}
    lat = combine(lattice.(hs)...)
    subsizes = sublatlengths(lat)
    blocksizes = Int[]
    for h in hs
        append!(blocksizes, norbitals(h))
    end
    bs = OrbitalBlockStructure{B}(blocksizes, subsizes)
    builder = IJVBuilder(lat, bs)
    blockoffsets = lengths_to_offsets(size.(hs, 1))
    for (i, h) in enumerate(hs)
        pushblock!(builder, h, blockoffsets[i])
    end
    assigned = Matrix{Bool}(I, length(hs), length(hs))
    return Coupler(hs, lat, builder, blockoffsets, assigned)
end

function pushblock!(builder, h, offset)
    for hh in harmonics(h)
        dn = dcell(hh)
        ijv = builder[dn]
        is, js, vs = findnz(unflat(matrix(hh)))
        if !iszero(offset)
            is .+= offset
            js .+= offset
        end
        append!(ijv, (is, js, vs))
    end
    return builder
end

function coupleblock!(c::Coupler, model::TightbindingModel, ib, jb)
    lat = lattice(c)
    orbstruct = blockstructure(c)
    amodel = apply(model, (lat, orbstruct))
    b = builder(c)
    irng, jrng = blockranges(c, ib, jb)
    assignedblocks(c)[ib, jb] && deleteblock!(b, irng, jrng)
    foreach(term -> applyterm!(b, term, (irng, jrng)), terms(amodel))
    assignedblocks(c)[ib, jb] = true
    return c
end

deleteblock!(builder, irng, jrng) = filter!((i, j, v) -> !(i in irng && j in jrng), builder)

function hamiltonian(c::Coupler)
    hars = sparse(builder(c))
    lat = lattice(c)
    orbstruct = blockstructure(c)
    return Hamiltonian(lat, orbstruct, hars)
end

function parametric(c::Coupler)
    h = hamiltonian(c)
    modifiers = tuplejoin(reapply_modifiers.(Ref(h), blocks(c), Base.front(blockoffsets(c)))...)
    return parametric(h, modifiers...)
end

### Recompute pointers of modifiers for new block Hamiltonian

reapply_modifiers(hnew, h::Hamiltonian, offset) = ()
reapply_modifiers(hnew, h::ParametricHamiltonian, offset) =
    _reapply_modifiers(hnew, h, offset, modifiers(h)...)

_reapply_modifiers(hnew, h, offset) = ()
_reapply_modifiers(hnew, h, offset, m, ms...) =
    (reapply_modifier(hnew, h, offset, m), _reapply_modifiers(hnew, h, offset, ms...)...)

function reapply_modifier(hnew, hold, offset, m::AppliedOnsiteModifier)
    m´ = similar(m)
    ps, ps´ = pointers(m), pointers(m´)
    har0, har0´ = first(harmonics(hold)), first(harmonics(hnew))
    h, h´ = unflat(matrix(har0)), unflat(matrix(har0´))
    update_ptrs!(ps´, ps, h´, h, offset)
    return m´
end

function reapply_modifier(hnew, hold, offset, m::AppliedHoppingModifier)
    m´ = similar(m)
    hars, hars´ = harmonics(hold), harmonics(hnew)
    emptyptrs!(m´, length(hars´))
    pss, pss´ = pointers(m), pointers(m´)
    for (ps´, har´) in zip(pss´, hars´), (ps, har) in zip(pss, hars)
        if dcell(har´) == dcell(har)
            h, h´ = unflat(matrix(har)), unflat(matrix(har´))
            update_ptrs!(ps´, ps, h´, h, offset)
        end
    end
    return m´
end

function update_ptrs!(ps´, ps, h´, h, offset)
    col = 1
    for (ptr, rest...) in ps
        found = false
        row = rowvals(h)[ptr]
        col = findcol(h, col, ptr)
        col´ = col + offset
        for ptr´ in nzrange(h´, col´)
            row´ = rowvals(h´)[ptr´]
            if row´ == row + offset
                push!(ps´, (ptr´, rest...))
                found = true
                break
            end
        end
        found || internalerror("reapply_modifier")
    end
    return ps´
end

function findcol(h0, col, ptr)
    for col´ in col:size(h0, 2)
        ptr in nzrange(h0, col´) && return col´
    end
    return size(h0, 2)
end

#endregion