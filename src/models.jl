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
    ParametricModel(ParametricOnsiteTerm(functor(o), siteselector(selector(o); kw...), coefficient(o)))
_hopping(t::ParametricHoppingTerm; kw...) =
    ParametricModel(ParametricHoppingTerm(functor(t), hopselector(selector(t); kw...), coefficient(t)))

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

# version with site selector kwargs
macro onsite(kw, f)
    f, N, params = get_f_N_params(f, "Only @onsite(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.ParametricModel(Quantica.ParametricOnsiteTerm(
        Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector($kw), 1))))
end

# version without site selector kwargs
macro onsite(f)
    f, N, params = get_f_N_params(f, "Only @onsite(args -> body; kw...) syntax supported.  Mind the `;`.")
    return esc(:(Quantica.ParametricModel(Quantica.ParametricOnsiteTerm(
            Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector(), 1))))
end

# version with hop selector kwargs
macro hopping(kw, f)
    f, N, params = get_f_N_params(f, "Only @hopping(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.ParametricModel(Quantica.ParametricHoppingTerm(
        Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector($kw), 1))))
end

# version without hop selector kwargs
macro hopping(f)
    f, N, params = get_f_N_params(f, "Only @hopping(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.ParametricModel(Quantica.ParametricHoppingTerm(
        Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector(), 1))))
end

## Model modifiers ##

macro onsite!(kw, f)
    f, N, params = get_f_N_params(f, "Only @onsite!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.OnsiteModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector($kw))))
end

macro onsite!(f)
    f, N, params = get_f_N_params(f, "Only @onsite!(args -> body; kw...) syntax supported.  Mind the `;`.")
    return esc(:(Quantica.OnsiteModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.siteselector())))
end

macro hopping!(kw, f)
    f, N, params = get_f_N_params(f, "Only @hopping!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector_unbounded($kw))))
end

macro hopping!(f)
    f, N, params = get_f_N_params(f, "Only @hopping!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector_unbounded())))
end

# Extracts normalized f, number of arguments and kwarg names from an anonymous function f
function get_f_N_params(f, msg)
    (f isa Expr && f.head == :->) || throw(ArgumentError(msg))
    d = ExprTools.splitdef(f)
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
    return f´, N, params
end

get_kwname(x::Symbol) = x
get_kwname(x::Expr) = x.head === :kw ? x.args[1] : x.head  # x.head == :...

#endregion

############################################################################################
# @onsite, @hopping conversions
#region

zero_model(term::ParametricOnsiteTerm) =
    OnsiteTerm(r -> 0I, selector(term), coefficient(term))
zero_model(term::ParametricHoppingTerm) =
    HoppingTerm((r, dr) -> 0I, selector(term), coefficient(term))

function modifier(term::ParametricOnsiteTerm{N}) where {N}
    f = (o, args...; kw...) -> o + term(args...; kw...)
    pf = ParametricFunction{N+1}(f, parameters(term))
    return OnsiteModifier(pf, selector(term))
end

function modifier(term::ParametricHoppingTerm{N}) where {N}
    f = (t, args...; kw...) -> t + term(args...; kw...)
    pf = ParametricFunction{N+1}(f, parameters(term))
    return HoppingModifier(pf, selector(term))
end

basemodel(m::ParametricModel) = nonparametric(m) + TightbindingModel(zero_model.(terms(m)))

# transforms the first argument in each model term to a parameter named pname
model_ω_to_param(model::ParametricModel) =
    ParametricModel(nonparametric(model), model_ω_to_param.(terms(model)))

model_ω_to_param(model::TightbindingModel) = model_ω_to_param(ParametricModel(model))

function model_ω_to_param(term::ParametricOnsiteTerm{N}, default = 0) where {N}
    # parameters(term) only needed for reporting, we omit adding :ω_internal
    f = (args...; ω_internal = default, kw...) -> term(ω_internal, args...; kw...)
    pf = ParametricFunction{N-1}(f, parameters(term))
    return ParametricOnsiteTerm(pf, selector(term), coefficient(term))
end

function model_ω_to_param(term::ParametricHoppingTerm{N}, default = 0) where {N}
    # parameters(term) only needed for reporting, we omit adding :ω_internal
    f = (args...; ω_internal = default, kw...) -> term(ω_internal, args...; kw...)
    pf = ParametricFunction{N-1}(f, parameters(term))
    return ParametricHoppingTerm(pf, selector(term), coefficient(term))
end

#endregion

############################################################################################
# interblock and intrablock
#region

interblock(m::AbstractModel, hams::AbstractHamiltonian...) =
    interblock(m, blockindices(hams)...)

interblock(m::AbstractModel, inds...) = isempty(intersect(inds...)) ?
    InterblockModel(hopping(m), inds) :
    InterblockModel(m, inds)                 # if blocks overlap, don't exclude onsite terms

intrablock(m::AbstractModel, inds) = IntrablockModel(m, inds)

function blockindices(hams::NTuple{N,<:Any}) where {N}
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