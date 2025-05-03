############################################################################################
# onsite and hopping
#region

onsite(o; kw...) = onsite(o, siteselector(; kw...))
onsite(o, sel::SiteSelector) = TightbindingModel(OnsiteTerm(o, sel, 1))

hopping(t; kw...) = hopping(t, hopselector(; kw...))
hopping(t, sel::HopSelector) = TightbindingModel(HoppingTerm(t, sel, 1))

plusadjoint(t) = t + t'

## filtering models, and modifying their selectors

onsite(m::AbstractModel; kw...) =
    reduce(+, (_onsite(t; kw...) for t in allterms(m) if t isa Union{OnsiteTerm,ParametricOnsiteTerm}); init = TightbindingModel())
hopping(m::AbstractModel; kw...) =
    reduce(+, (_hopping(t; kw...) for t in allterms(m) if t isa Union{HoppingTerm,ParametricHoppingTerm}); init = TightbindingModel())

_onsite(o::OnsiteTerm; kw...) =
    TightbindingModel(OnsiteTerm(functor(o), siteselector(selector(o); kw...), coefficient(o)))
_hopping(t::HoppingTerm; kw...) =
    TightbindingModel(HoppingTerm(functor(t), hopselector(selector(t); kw...), coefficient(t)))
_onsite(o::ParametricOnsiteTerm; kw...) =
    ParametricModel(ParametricOnsiteTerm(functor(o), siteselector(selector(o); kw...), coefficient(o), is_spatial(o)))
_hopping(t::ParametricHoppingTerm; kw...) =
    ParametricModel(ParametricHoppingTerm(functor(t), hopselector(selector(t); kw...), coefficient(t), is_spatial(t)))

#endregion

############################################################################################
# @onsite, @hopping, @onsite! and @hopping! - Parametric models and model modifiers
#region

# Macros are needed to read out number of f arguments (N) and kwarg names (params).
# A kw... is appended to kwargs in the actual function method definition to skip
# non-applicable kwargs.
# An alternative based on internals (m = first(methods(f)), Base.kwarg_decl(m) and m.nargs)
# has been considered, but decided against due to its fragility and slow runtime

## Parametric models ##

macro onsite(x, ys...)
    kw, f, N, params, spatial = parse_term("@onsite", x, ys...)
    return esc(:(Quantica.ParametricModel(Quantica.ParametricOnsiteTerm(
        Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector($kw), 1, $(spatial)))))
end

macro hopping(x, ys...)
    kw, f, N, params, spatial = parse_term("@hopping", x, ys...)
    return esc(:(Quantica.ParametricModel(Quantica.ParametricHoppingTerm(
        Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector($kw), 1, $(spatial)))))
end

## Model modifiers ##

macro onsite!(x, ys...)
    kw, f, N, params, spatial = parse_term("@onsite!", x, ys...)
    return esc(:(Quantica.OnsiteModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector($kw), $(spatial))))
end

macro hopping!(x, ys...)
    kw, f, N, params, spatial = parse_term("@hopping!", x, ys...)
    # Since the default hopping range is neighbors(1), we need change the default to Inf for @hopping!
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector_infrange($kw), $(spatial))))
end

function parse_term(macroname, x, ys...)
    if x isa Expr && x.head == :parameters
        kw = x
        f, N, params, spatial = parse_term_body(macroname, only(ys))
    # elseif x isa Expr && x.head == :block  # received a quoted expression
    #     dump(x)
    #     return parse_term(macroname, only(x.args), ys...)
    else
        kw = parse_term_parameters(ys...)
        f, N, params, spatial = parse_term_body(macroname, x)
    end
    return kw, f, N, params, spatial
end

# parse keywords after a comma as if they were keyword arguments
function parse_term_parameters(exs...)
    exs´ = maybe_kw.(exs)
    paramex = Expr(:parameters, exs´...)
    return paramex
end

maybe_kw(ex::Expr) = Expr(:kw, ex.args...)
maybe_kw(ex::Symbol) = ex

# Extracts normalized f, number of arguments and kwarg names from an anonymous function f
function parse_term_body(macroname, f)
    if !(f isa Expr && (f.head == :-> || f.head == :-->))
        msg = "Only $(macroname)(args -> body; kw...) syntax supported (or with -->). Received $(macroname)($f, ...) instead."
        throw(ArgumentError(msg))
    end
    # change --> to -> and record change in spatial
    spatial = f.head == :->
    !spatial && (f.head = :->)
    d = ExprTools.splitdef(f)
    # process keyword arguments, add splat
    kwargs = convert(Vector{Any}, get!(d, :kwargs, []))
    d[:kwargs] = kwargs  # in case it wasn't Vector{Any} originally
    if isempty(kwargs)
        params = Symbol[]
        push!(kwargs, :(_...))  # normalization : append _... to kwargs
    else
        params = get_kwname.(kwargs)
        if !isempty(params) && last(params) == :...
            params = params[1:end-1]  # drop _... kwarg from params
        else
            push!(kwargs, :(_...))  # normalization : append _... to kwargs
        end
    end
    N = haskey(d, :args) ? length(d[:args]) : 0
    f´ = ExprTools.combinedef(d)
    return f´, N, params, spatial
end

get_kwname(x::Symbol) = x
get_kwname(x::Expr) = x.head === :kw ? x.args[1] : x.head  # x.head == :...

hopselector_infrange(; kw...) = hopselector(; range = Inf, kw...)

#endregion

############################################################################################
# @onsite, @hopping conversions
#   each ParametricTerm gets converted, using `basemodel` and `modifier`, into two pieces
#region

zero_model(term::ParametricOnsiteTerm) =
    OnsiteTerm(r -> 0I, selector(term), coefficient(term))
zero_model(term::ParametricHoppingTerm) =
    HoppingTerm((r, dr) -> 0I, selector(term), coefficient(term))

function modifier(term::ParametricOnsiteTerm{N}) where {N}
    f = (o, args...; kw...) -> o + term(args...; kw...)
    pf = ParametricFunction{N+1}(f, parameter_names(term))
    return OnsiteModifier(pf, selector(term), is_spatial(term))
end

function modifier(term::ParametricHoppingTerm{N}) where {N}
    f = (t, args...; kw...) -> t + term(args...; kw...)
    pf = ParametricFunction{N+1}(f, parameter_names(term))
    return HoppingModifier(pf, selector(term), is_spatial(term))
end

basemodel(m::ParametricModel) = nonparametric(m) + TightbindingModel(zero_model.(terms(m)))

# transforms the first argument in each model term to a parameter named pname
model_ω_to_param(model::ParametricModel) =
    ParametricModel(nonparametric(model), model_ω_to_param.(terms(model)))

model_ω_to_param(model::TightbindingModel) = model_ω_to_param(ParametricModel(model))

function model_ω_to_param(term::ParametricOnsiteTerm{N}, default = 0) where {N}
    # parameter_names(term) only needed for reporting, we omit adding :ω_internal
    f = (args...; ω_internal = default, kw...) -> term(ω_internal, args...; kw...)
    pf = ParametricFunction{N-1}(f, parameter_names(term))
    return ParametricOnsiteTerm(pf, selector(term), coefficient(term), is_spatial(term))
end

function model_ω_to_param(term::ParametricHoppingTerm{N}, default = 0) where {N}
    # parameter_names(term) only needed for reporting, we omit adding :ω_internal
    f = (args...; ω_internal = default, kw...) -> term(ω_internal, args...; kw...)
    pf = ParametricFunction{N-1}(f, parameter_names(term))
    return ParametricHoppingTerm(pf, selector(term), coefficient(term), is_spatial(term))
end

#endregion

############################################################################################
# interblock and intrablock
#region

interblock(m::AbstractModel, hams::AbstractHamiltonian...) =
    interblock(m, blockindices(hams)...)

interblock(m::AbstractModel, inds...) = isempty(intersect(inds...)) ?
    Interblock(hopping(m), inds) :
    Interblock(m, inds)                 # if blocks overlap, don't exclude onsite terms

intrablock(m::AbstractModel, inds) = Intrablock(m, inds)

interblock(m::AbstractModifier, inds) = Interblock(m, inds)
intrablock(m::AbstractModifier, inds) = Intrablock(m, inds)

function blockindices(hams::NTuple{N,Any}) where {N}
    offset = 0
    inds = ntuple(Val(N)) do i
        ns = nsites(lattice(hams[i]))
        rng = offset + 1:offset + ns
        offset += ns
        rng
    end
    return inds
end

#endregion

############################################################################################
# Pauli matrices
#region

const pauli_names = (;
    x = 1, y = 2, z = 3, I = 0,
    II = (0,0), Ix = (0,1), Iy = (0,2), Iz = (0,3),
    xI = (1,0), xx = (1,1), xy = (1,2), xz = (1,3),
    yI = (2,0), yx = (2,1), yy = (2,2), yz = (2,3),
    zI = (3,0), zx = (3,1), zy = (3,2), zz = (3,3))

function σ(n::Integer)
    if n == 0
        return SA[1.0 + 0im 0; 0 1]
    elseif n == 1
        return SA[0.0 + 0im 1; 1 0]
    elseif n == 2
        return SA[0.0 + 0im -im; im 0]
    elseif n == 3
        return SA[1.0 + 0im 0; 0 -1]
    else
        argerror("Pauli matrix index $n out of range (0-3)")
    end
end

σ((θ, ϕ)::Tuple{Real,Real}) = σ(1)*sin(θ)*cos(ϕ) + σ(2)*sin(θ)*sin(ϕ) + σ(3)*cos(θ)
σ(v::SVector{3}) = dot(normalize(v), SA[σ(1), σ(2), σ(3)])

σ(n1, n2, ns...) = kron(σ(n1), σ(n2), σ.(ns)...)

function σ(s::Symbol)
    if haskey(pauli_names, s)
        return σ(pauli_names[s]...)
    else
        argerror("Unrecognized Pauli matrix identifier $s, should be one of $(keys(pauli_names))")
    end
end

Base.getproperty(::typeof(σ), s::Symbol) = σ(s)

#endregion
