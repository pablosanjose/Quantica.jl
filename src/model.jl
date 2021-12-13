############################################################################################
# Model terms constructors
#region

TightbindingModel(ts::TightbindingModelTerm...) = TightbindingModel(ts)

OnsiteTerm(t::OnsiteTerm, os::SiteSelector) = OnsiteTerm(t.o, os, t.coefficient)

HoppingTerm(t::HoppingTerm, os::HopSelector) = HoppingTerm(t.t, os, t.coefficient)

onsite(o; kw...) = onsite(o, siteselector(; kw...))
onsite(o, sel::SiteSelector) = TightbindingModel(OnsiteTerm(o, sel, 1))
onsite(m::TightbindingModel; kw...) = TightbindingModel(
    Tuple(Any[OnsiteTerm(o, siteselector(o.selector; kw...)) for o in terms(m) if o isa OnsiteTerm]))

function hopping(t; plusadjoint = false, kw...)
    hop = hopping(t, hopselector(; kw...))
    return plusadjoint ? hop + hop' : hop
end

hopping(t, sel::HopSelector) = TightbindingModel(HoppingTerm(t, sel, 1))
hopping(m::TightbindingModel; kw...) = TightbindingModel(
    Tuple(Any[HoppingTerm(t, hopselector(t.selector; kw...)) for t in terms(m) if t isa HoppingTerm]))

#endregion

############################################################################################
# Model terms algebra
#region

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* terms(m))
Base.:*(m::TightbindingModel, x::Number) = x * m
Base.:-(m::TightbindingModel) = (-1) * m

Base.:+(m::TightbindingModel, m´::TightbindingModel) = TightbindingModel((terms(m)..., terms(m´)...))
Base.:-(m::TightbindingModel, m´::TightbindingModel) = m + (-m´)

Base.:*(x::Number, o::OnsiteTerm) = OnsiteTerm(o.o, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.t, t.selector, x * t.coefficient)

Base.adjoint(m::TightbindingModel) = TightbindingModel(adjoint.(terms(m)))
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.o(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.o', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.t(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.t', t.selector', t.coefficient')

#endregion

############################################################################################
# Model modifiers constructors
#region

# Modifiers need macros to read out number of f arguments (N) and kwarg names (params).
# A kw... is appended to kwargs in the actual function method definition to skip
# non-applicable kwargs.
# An alternative based on internals (m = first(methods(f)), Base.kwarg_decl(m) and m.nargs)
# has been considered, but decided against due to its fragility and slow runtime

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
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector($kw))))
end

macro hopping!(f)
    f, N, params = get_f_N_params(f, "Only @hopping!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(params)), Quantica.hopselector())))
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
