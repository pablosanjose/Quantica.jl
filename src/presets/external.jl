############################################################################################
# ExternalPresets
#region

module ExternalPresets

using Quantica

############################################################################################
# WannierBuilder
#region

using Quantica: IJVBuilder, IJVHarmonic, IJV, AbstractHamiltonianBuilder, AbstractModel,
    BarebonesOperator, Modifier, push_ijvharmonics!, postype, tupleflatten, dcell, ncells,
    collector, builder, modifiers, harmonics

struct WannierBuilder{T,L} <: AbstractHamiltonianBuilder{T,L,L,Complex{T}}
    hbuilder::IJVBuilder{T,L,L,Complex{T},Vector{Any}}
    rharmonics::Vector{IJVHarmonic{L,SVector{L,Complex{T}}}}
end

struct Wannier90Data{T,L}
    brvecs::NTuple{L,SVector{L,T}}
    norbs::Int
    ncells::Int
    h::Vector{IJVHarmonic{L,Complex{T}}}             # must be dn-sorted, with (0,0,0) first
    r::Vector{IJVHarmonic{L,SVector{L,Complex{T}}}}  # must be dn-sorted, with (0,0,0) first
end

#region ## Constructors ##

function WannierBuilder(file::AbstractString; kw...)
    data = load_wannier90(file; kw...)
    lat = lattice(data)
    hbuilder = builder(lat; orbitals = Val(1)) # one orbital per site
    push_ijvharmonics!(hbuilder, data.h)
    rharmonics = data.r
    return WannierBuilder(hbuilder, rharmonics)
end

function WannierBuilder(file, model::AbstractModel; kw...)
    b = WannierBuilder(file; kw...)
    add!(b.hbuilder, model)
    return b
end

(m::Quantica.Modifier)(b::WannierBuilder) = (push!(b, m); b)

#endregion

#region ## API ##

wannier90(args...; kw...) = WannierBuilder(args...; kw...)    # API (extendable)

Quantica.hamiltonian(b::WannierBuilder) = hamiltonian(b.hbuilder)

Quantica.position(b::WannierBuilder) = BarebonesOperator(b.rharmonics)

Quantica.ncells(b::WannierBuilder) = ncells(b.hbuilder)

Quantica.modifiers(b::WannierBuilder) = modifiers(b.hbuilder)

Quantica.harmonics(b::WannierBuilder) = harmonics(b.hbuilder)

Quantica.lattice(b::WannierBuilder) = lattice(b.hbuilder)

Quantica.blockstructure(b::WannierBuilder) = blockstructure(b.hbuilder)

Base.push!(b::WannierBuilder, m::Modifier) = push!(b.hbuilder, m)

Base.pop!(b::WannierBuilder) = pop!(b.hbuilder)

hbuilder(b::WannierBuilder) = b.hbuilder

nelements(b::WannierBuilder) = sum(length, harmonics(b))

#endregion

#region ## Wannier90Data ##

load_wannier90(filename; type = Float64, dim = 3, kw...) =
    load_wannier90(filename, postype(dim, type); kw...)

function load_wannier90(filename, ::Type{SVector{L,T}}; htol = 1e-8, rtol = 1e-8) where {L,T}
    L > 3 && argerror("dim = $L should be dim <= 3")
    data = open(filename, "r") do f
        # skip header
        readline(f)
        # read three Bravais 3D-vectors, keep L Bravais LD-vectors
        brvecs3 = ntuple(_ -> SVector{3,T}(readline_realtypes(f, SVector{3,T})), Val(3))
        brvecs  = ntuple(i -> brvecs3[i][SVector{L,Int}(1:L)], Val(L))
        # read number of orbitals
        norbs = readline_realtypes(f, Int)
        # read number of cells
        ncells = readline_realtypes(f, Int)
        # skip symmetry degeneracies
        while !eof(f)
            isempty(readline(f)) && break
        end
        # read Hamiltonian
        h = load_harmonics(f, Val(L), Complex{T}, norbs, ncells, htol)
        # read positions
        r = load_harmonics(f, Val(L), SVector{L,Complex{T}}, norbs, ncells, rtol)
        return Wannier90Data(brvecs, norbs, ncells, h, r)
    end
    return data
end

function load_harmonics(f, ::Val{L}, ::Type{B}, norbs, ncells, atol) where {L,B}
    ncell = 0
    hars = IJVHarmonic{L,B}[]
    ijv = IJV{B}(norbs^2)
    while !eof(f)
        ncell += 1
        dn3D = SVector{3,Int}(readline_realtypes(f, SVector{3,Int}))
        dn = dn3D[SVector{L,Int}(1:L)]
        # if skip, read but not write harmonic (since it is along a non-projected axes)
        skip = L < 3 && !iszero(dn3D[SVector{3-L,Int}(L+1:3)])
        for j in 1:norbs, i in 1:norbs
            i´, j´, reims... = readline_realtypes(f, Int, Int, B)
            skip && continue
            i´ == i && j´ == j ||
                argerror("load_wannier90: unexpected entry in file at element $((dn, i, j))")
            v = build_complex(reims, B)
            push_if_nonzero!(ijv, (i, j, v), dn, atol)
        end
        if !skip && !isempty(ijv)
            push!(hars, IJVHarmonic(dn, ijv))
            ijv = IJV{B}(norbs^2)    # allocate new IJV
        end
        isempty(readline(f)) ||
            argerror("load_wannier90: unexpected line after harmonic $dn")
        ncell == ncells && break
    end
    sort!(hars, by = ijv -> abs.(dcell(ijv)))
    iszero(dcell(first(hars))) ||
        pushfirst!(hars, IJVHarmonic(zero(SVector{L,Int}), IJV{B}()))
    return hars
end

readline_realtypes(f, type::Type{<:Real}) = parse(type, readline(f))

readline_realtypes(f, types::Type...) =
    readline_realtypes(f, tupleflatten(realtype.(types)...)...)

function readline_realtypes(f, realtypes::Vararg{<:Type{<:Real},N}) where {N}
    tokens = split(readline(f))
    # realtypes could be less than originally read tokens if we have reduced dimensionality
    reals = ntuple(i -> parse(realtypes[i], tokens[i]), Val(N))
    return reals
end

realtype(t::Type{<:Real}) = t
realtype(::Type{Complex{T}}) where {T} = (T, T)
realtype(::Type{SVector{N,T}}) where {N,T<:Real} = ntuple(Returns(T), Val(N))
realtype(::Type{SVector{N,Complex{T}}}) where {N,T<:Real} =
    (t -> (t..., t...))(realtype(SVector{N,T}))

build_complex((r, i), ::Type{B}) where {B<:Complex} = Complex(r, i)
build_complex(ri, ::Type{B}) where {C<:Complex,N,B<:SVector{N,C}} =
    SVector{N,C}(ntuple(i -> C(ri[2i-1], ri[2i]), Val(N)))

push_if_nonzero!(ijv::IJV{<:Number}, (i, j, v), dn, htol) =
    abs(v) > htol && push!(ijv, (i, j, v))
push_if_nonzero!(ijv::IJV{<:SVector}, (i, j, v), dn, rtol) =
    (i == j && iszero(dn) || any(>(rtol), abs.(v))) && push!(ijv, (i, j, v))

function Quantica.lattice(data::Wannier90Data{T,L}) where {T,L}
    bravais = hcat(data.brvecs...)
    rs = SVector{L,T}[]
    ijv = collector(first(data.r))
    for (i, j, r) in zip(ijv.i, ijv.j, ijv.v)
        i == j && push!(rs, real(r))
    end
    return lattice(sublat(rs); bravais)
end

#endregion

#region ## show ##

function Base.show(io::IO, b::WannierBuilder)
    i = get(io, :indent, "")
    print(io, i, summary(b), "\n",
"$i  cells      : $(ncells(b))
$i  elements   : $(nelements(b))
$i  modifiers  : $(length(modifiers(b)))")
end

Base.summary(::WannierBuilder{T,L}) where {T,L} =
    "WannierBuilder{$T,$L} : $(L)-dimensional Hamiltonian builder from Wannier90 input of type $T"

#endregion

#endregion

end # module

const EP = ExternalPresets

#endregion
