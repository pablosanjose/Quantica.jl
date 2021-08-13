############################################################################################
# Model constructors
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

hopping(t, sel::Selector) = TightbindingModel(HoppingTerm(t, sel, 1))
hopping(m::TightbindingModel; kw...) = TightbindingModel(
    Tuple(Any[HoppingTerm(t, hopselector(t.selector; kw...)) for t in terms(m) if t isa HoppingTerm]))

#endregion

############################################################################################
# Model call and apply
#region

(o::OnsiteTerm{<:Function})(r,dr) = o.coefficient * o.o(r)
(o::OnsiteTerm)(r,dr) = o.coefficient * o.o

(h::HoppingTerm{<:Function})(r, dr) = h.coefficient * h.t(r, dr)
(h::HoppingTerm)(r, dr) = h.coefficient * h.t

apply(t::OnsiteTerm, lat::Lattice) = OnsiteTerm(t.o, apply(t.selector, lat), t.coefficient)
apply(t::HoppingTerm, lat::Lattice) = HoppingTerm(t.t, apply(t.selector, lat), t.coefficient)
apply(m::TightbindingModel, lat::Lattice) = TightbindingModel(apply.(terms(m), Ref(lat)))

#endregion

############################################################################################
# Model algebra
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