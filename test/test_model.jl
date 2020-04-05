using Quantica: TightbindingModel, OnsiteTerm, HoppingTerm, padtotype, Selector

@testset "onsite" begin
    r = SVector(0.0, 0.0)
    os = (I, 2I, 2.0I,@SMatrix[1 0; 0 1], r -> 2.0I, r -> I, r -> SMatrix{3,3}(3I))
    ss = (missing, :A, (:A,), (:A, :B))
    cs = (1, 1.0, 1.0f0)
    ts = (Float32, Float64, SMatrix{3,3}, SMatrix{3,2,Float64}, SMatrix{1,4,Float32})
    for o in os, s in ss, c in cs
        ons = c * onsite(o, sublats = s)
        @test ons isa
            TightbindingModel{1,<:Tuple{OnsiteTerm{typeof(o)}}}
        term = first(ons.terms)
        for t in ts
            @test padtotype(term(r, r), t) isa t
        end
    end
end

@testset "hopping" begin
    r = SVector(0.0, 0.0)
    hs = (I, 2I, 2.0I, @SMatrix[1 0; 0 1], (r, dr) -> 2.0I, (r, dr) -> I, (r, dr) -> SMatrix{3,3}(3I))
    ss = (missing, :A, (:A,), (:A, :B), ((:A,:B), :C), ((:A,:B), (:C,)), ((:A,:B), (:C, :D)))
    cs = (1, 1.0, 1.0f0)
    ts = (Float32, Float64, SMatrix{3,3}, SMatrix{3,2,Float64}, SMatrix{1,4,Float32})
    dns = (missing, (1,), ((1,2), (3,4)))
    rs = (1, 2.0, Inf)
    for h in hs, s in ss, c in cs, d in dns, r in rs
        hop = c * hopping(h, sublats = s, dn = d, range = r)
        @test hop isa
            TightbindingModel{1,<:Tuple{HoppingTerm{typeof(h)}}}
        term = first(hop.terms)
        for t in ts
            @test padtotype(term(r, r), t) isa t
        end
    end
end

@testset "term algebra" begin
    r = SVector(0, 0)
    model = onsite(1) + hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (1, 2I)
    model = onsite(1) - hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (1, -2I)
    model = -onsite(@SMatrix[1 0; 1 1]) - 2hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (-@SMatrix[1 0; 1 1], -4I)
    @test model(r, r) == @SMatrix[-5 0; -1 -5]
end