using Quantica: TightbindingModel, OnsiteTerm, HoppingTerm, padtotype, Selector, sublats

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

@testset "onsite terms" begin
    rs = (r->true, missing)
    ss = (:A, (:A, :B), missing)
    for r in rs, s in ss
        model0 = onsite(1, region = r, sublats = s) + hopping(1)
        model1 = onsite(1) + hopping(1)
        model2 = onsite(1, sublats = s) + hopping(1)
        model3 = onsite(1, region = r) + hopping(1)
        @test onsite(model1, region = r, sublats = s) + hopping(model1) === model0
        @test onsite(model2, region = r, sublats = s) + hopping(model2) === model0
        @test onsite(model3, region = r, sublats = s) + hopping(model3) === model0
    end
end

@testset "hopping terms" begin
    rs = ((r, dr) -> true, missing)
    ss = (:A => :B, :A => (:A,:B), (:A,:B) .=> (:A,:B), (:A,:B) => (:A,:B), missing)
    dns = ((0,1), ((0,1),(1,0)), SVector(0,1), (SVector(0,1), (0,3)), [1, 2], ([1.0,2], (0,4.0)), missing)
    ranges = (Inf, 1)  # no missing here, because hopping range default is 1.0
    for r in rs, s in ss, dn in dns, rn in ranges
        model0 = hopping(1, region = r, sublats = s, dn = dn, range = rn) + onsite(1)
        model1 = hopping(1, region = r, sublats = s, dn = dn) + onsite(1)
        model2 = hopping(1, region = r, sublats = s) + onsite(1)
        model3 = hopping(1, region = r, range = rn) + onsite(1)
        model4 = hopping(1) + onsite(1)
        @test hopping(model1, region = r, sublats = s, dn = dn, range = rn) + onsite(model1) === model0
        @test hopping(model2, region = r, sublats = s, dn = dn, range = rn) + onsite(model2) === model0
        @test hopping(model3, region = r, sublats = s, dn = dn, range = rn) + onsite(model3) === model0
        @test hopping(model4, region = r, sublats = s, dn = dn, range = rn) + onsite(model4) === model0
    end

    sublats(hopping(1, sublats = (:A,:B) => (:A,:B))) == ((:A => :A, :A => :B, :B => :A, :B => :B),)
    sublats(hopping(1, sublats = (:A,:B) .=> (:A,:B))) == ((:A => :A, :B => :B),)
    sublats(hopping(1, sublats = :A => (:A,:B))) == ((:A => :A, :A => :B),)
    sublats(hopping(1, sublats = (:C => :C, (:A,:B) => (:A,:B)))) == ((:C => :C, :A => :A, :A => :B, :B => :A, :B => :B),)
end

@testset "hopping adjoint" begin
    ts  = (1, 1im, @SMatrix[0 -im; im 0], @SMatrix[0 -im])
    ss  = (:A => :B, :A => (:A,:B), (:A,:B) .=> (:A,:B), (:A,:B) => (:A,:B), missing)
    ss´ = (:B => :A, (:A,:B) => :A, (:A,:B) .=> (:A,:B), (:A,:B,:A,:B) .=> (:A,:A,:B,:B), missing)
    dns = ((0,1), ((0,1),(1,0)),    SVector(0,1),  (SVector(0,1), (0,3)),    [1, 2], ([1.0,2], (0,4.0)),   missing)
    dns´= ((0,-1), ((0,-1),(-1,0)), SVector(0,-1), (SVector(0,-1), (0,-3)), -[1, 2], (-[1.0,2], (0,-4.0)), missing)
    ranges = (Inf, 1)
    for (t,t´) in zip(ts, adjoint.(ts)), (s, s´) in zip(ss, ss´), (dn, dn´) in zip(dns, dns´), rn in ranges
        hop = hopping(t, sublats = s, dn = dn, range = rn)
        hop´ = hopping(t´, sublats = s´, dn = dn´, range = rn)
        @test hop' === hop´
    end
end