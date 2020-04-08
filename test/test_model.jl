using Quantica: TightbindingModel, OnsiteTerm, HoppingTerm, padtotype, Selector

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

@testset "onsiteselector!" begin
    rs = (r->true, missing)
    ss = (:A, missing)
    for r in rs, s in ss
        model0 = onsite(1, region = r, sublats = s) + hopping(1)
        model1 = onsite(1) + hopping(1)
        model2 = onsite(1, sublats = s) + hopping(1)
        model3 = onsite(1, region = r) + hopping(1)
        @test onsiteselector!(model1, region = r, sublats = s) === model0
        @test onsiteselector!(model2, region = r, sublats = s) === model0
        @test onsiteselector!(model3, region = r, sublats = s) === model0
    end
end

@testset "hoppingselector!" begin
    rs = (r->true, missing)
    ss = (:A, missing)
    dns = ((0,1), missing)
    ranges = (Inf, 1)  # no missing here, because hopping range default is 1.0
    for r in rs, s in ss, dn in dns, rn in ranges
        model0 = hopping(1, region = r, sublats = s, dn = dn, range = rn) + onsite(1)
        model1 = hopping(1, region = r, sublats = s, dn = dn) + onsite(1)
        model2 = hopping(1, region = r, sublats = s) + onsite(1)
        model3 = hopping(1, region = r, range = rn) + onsite(1)
        model4 = hopping(1) + onsite(1)
        @test hoppingselector!(model1, region = r, sublats = s, dn = dn, range = rn) === model0
        @test hoppingselector!(model2, region = r, sublats = s, dn = dn, range = rn) === model0
        @test hoppingselector!(model3, region = r, sublats = s, dn = dn, range = rn) === model0
        @test hoppingselector!(model4, region = r, sublats = s, dn = dn, range = rn) === model0
    end
end