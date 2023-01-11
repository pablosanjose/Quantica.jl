############################################################################################
# Misc tools
#region

rdr((r1, r2)::Pair) = (0.5 * (r1 + r2), r2 - r1)

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

padtuple(t, x, N) = ntuple(i -> i <= length(t) ? t[i] : x, N)

@noinline internalerror(func::String) =
    throw(ErrorException("Internal error in $func. Please file a bug report at https://github.com/pablosanjose/Quantica.jl/issues"))

@noinline argerror(msg) = throw(ArgumentError(msg))

@noinline checkblocksize(::UniformScaling, s) = nothing
@noinline checkblocksize(val, s) = (size(val, 1), size(val, 2)) == s ||
    throw(ArgumentError("Expected an block or matrix of size $s, got size $((size(val, 1), size(val, 2)))"))

function boundingbox(positions)
    isempty(positions) && argerror("Cannot find bounding box of an empty collection")
    posmin = posmax = first(positions)
    for pos in positions
        posmin = min.(posmin, pos)
        posmax = max.(posmax, pos)
    end
    return (posmin, posmax)
end

deleteif!(test, v::AbstractVector) = deleteat!(v, (i for (i, x) in enumerate(v) if test(x)))

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameters(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

typename(x::T) where {T} = nameof(T)

# function get_or_push!(by, x, xs)
#     for x´ in xs
#         by(x) == by(x´) && return x´
#     end
#     push!(xs, x)
#     return x
# end

#endregion

############################################################################################
# Dynamic package loader
#   This is in global Quantica scope to avoid name collisions
#   We also `import` instead of `using` to avoid collisions between several backends
#region

function ensureloaded(package::Symbol)
    if !isdefined(Quantica, package)
        @warn("Required package $package not loaded. Loading...")
        eval(:(import $package))
    end
    return nothing
end

#endregion
