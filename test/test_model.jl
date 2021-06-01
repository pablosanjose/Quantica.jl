using Quantica: TightbindingModel, OnsiteTerm, HoppingTerm, padtotype, Selector, sublats, resolve, isinindices, isinsublats, nsites
using LinearAlgebra: norm

@testset "selector membership" begin
    @test  isinindices(1, missing)
    @test  isinindices(1, 1)
    @test !isinindices(2, 1)
    @test  isinindices(1, (1,2))
    @test !isinindices(3, (1,2))
    @test  isinindices(3, (1, 3:4))
    @test !isinindices(2, (1, 3:4))
    @test !isinindices(3, [1:2, 5:6])

    @test !isinindices(1, not(missing))
    @test !isinindices(1, not(1))
    @test  isinindices(2, not(1))
    @test !isinindices(1, not(1,2))
    @test  isinindices(3, not(1,2))
    @test !isinindices(3, not(1, 3:4))
    @test  isinindices(2, not(1, 3:4))
    @test  isinindices(3, not([1:2, 5:6]))

    @test  isinsublats(3, (not(1,2), 3))
    @test  isinsublats(3, (not(1,2), 4))
    @test !isinsublats(2, (not(1,2), 3))
    @test  isinsublats(3, (not(1,2), 3))

    @test  isinindices(1=>2, 1=>2)
    @test !isinindices(1=>2, 1=>3)
    @test  isinindices(1=>2, (1=>2, 3=>4))
    @test !isinindices(1=>2, (1=>3, 2=>1))
    @test  isinindices(1=>3, 1:2=>3:4)
    @test !isinindices(1=>2, 1:2=>3:4)
    @test  isinindices(1=>2, (1:2=>3:4, 1=>2))
    @test !isinindices(3=>4, (1:2=>3:4, 1=>2))
    @test  isinindices(1=>2, (1:2=>3:4, (1,2) .=> (2,1)))
    @test  isinindices(1=>3, (1:2=>3:4, 1:2 .=> (2,1)))
    @test !isinindices(3=>4, (1:2=>3:4, 1:2 .=> (2,1), 4=>3))

    @test !isinindices(1=>2, not(1=>2))
    @test  isinindices(1=>2, not(1=>3))
    @test !isinindices(1=>2, not(1=>2, 3=>4))
    @test  isinindices(1=>2, not(1=>3, 2=>1))
    @test !isinindices(1=>3, not(1:2=>3:4))
    @test  isinindices(1=>2, not(1:2=>3:4))
    @test !isinindices(1=>2, not(1:2=>3:4, 1=>2))
    @test  isinindices(3=>4, not(1:2=>3:4, 1=>2))
    @test !isinindices(1=>2, not(1:2=>3:4, (1,2) .=> (2,1)))
    @test !isinindices(1=>3, not(1:2=>3:4, 1:2 .=> (2,1)))
    @test  isinindices(3=>4, not(1:2=>3:4, 1:2 .=> (2,1), 4=>3))

    @test  isinindices(3=>4, not(1:2)=>3:4)
    @test !isinindices(1=>4, (not(1:2)=>3:4, 1:2 .=> (2,1), 4=>3))
    @test  isinindices(3=>4, (not(1:2)=>3:4, 1:2 .=> (2,1), 4=>3))
end

@testset "selector usage" begin
    lat = LatticePresets.honeycomb() |> unitcell(2)
    rs = (r->true, missing)
    ss = (:A, (:A, :B), missing)
    inds = (1, (1, 3), (1, 3:4), missing)
    for r in rs, s in ss, i in inds
        sel = siteselector(region = r, sublats = s, indices = i)
        rsel = resolve(sel, lat)
        @test 1 in rsel
        if s === :A || i !== missing
            @test !in(5, rsel)
        end
    end

    lat = LatticePresets.honeycomb() |> unitcell(2)
    rs =   ((r,dr)->true, missing)
    ss =   (:A=>:B, (:A=>:B, :B=>:C), (:A,:B)=>(:B,:A), ((:A,:B)=>(:B,:A), :A=>:C), ((:A,:B)=>(:B,:A), (:C,:D).=>(:C,:D)), ((:A,)=>(:A,), (:C,:D).=>(:C,:D), :C=>:C), missing)
    inds = (1=>2, (1=>2, 1=>4), 1:2=>3:4, (1:2=>3:4, 1=>2), (1:2=>3:4, (1,2) .=> (2,1)), (1:2=>3:4, 1:2 .=> (2,1), 4=>3), missing)
    for r in rs, s in ss, i in inds
        sel = hopselector(region = r, sublats = s, indices = i)
        rsel = resolve(sel, lat)
        conds = s in (:A=>:B, (:A=>:B, :B=>:C))
        condi = i in (1=>2,)
        if (conds !== missing && conds) || (condi !== missing && condi)
            @test !in(1=>4, rsel)
        else
            @test in(1=>4, rsel)
        end
    end

    h  = hamiltonian(lat, hopping(1, sublats = not(:A => :B)))
    h´ = hamiltonian(lat, hopping(1, sublats = :A => :B))
    @test !ishermitian(h) && !ishermitian(h´)
    @test bloch(h) == transpose(bloch(h´))
end

@testset "term algebra" begin
    r = SVector(0, 0)
    model = onsite(1) + hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (1, 2I)
    model = onsite(1) - hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (1, -2I)
    model = -onsite(@SMatrix[1 0; 1 1]) - 2hopping(2I)
    @test (t -> t(r, r)).(model.terms) == (-@SMatrix[1 0; 1 1], -4I)
end

@testset "onsite terms" begin
    rs = (r->true, missing)
    ss = (:A, (:A, :B), missing)
    inds = (1, (1, 2), (1:2, 3), missing)
    for r in rs, s in ss, i in inds
        model0 = onsite(1, region = r, sublats = s, indices = i) + hopping(1)
        model1 = onsite(1) + hopping(1)
        model2 = onsite(1, sublats = s) + hopping(1)
        model3 = onsite(1, region = r) + hopping(1)
        model4 = onsite(1, indices = i) + hopping(1)
        @test onsite(model1, region = r, sublats = s, indices = i) + hopping(model1) === model0
        @test onsite(model2, region = r, sublats = s, indices = i) + hopping(model2) === model0
        @test onsite(model3, region = r, sublats = s, indices = i) + hopping(model3) === model0
        @test onsite(model4, region = r, sublats = s, indices = i) + hopping(model3) === model0
    end
end

@testset "hopping terms" begin
    rs = ((r, dr) -> true, missing)
    ss = (:A => :B, :A => (:A,:B), (:A,:B) .=> (:A,:B), (:A,:B) => (:A,:B), missing)
    dns = ((0,1), ((0,1),(1,0)), SVector(0,1), (SVector(0,1), (0,3)), [1, 2], ([1.0,2], (0,4.0)), missing)
    inds = ((1 => 2, ), [1 => 2], (1 => 2, 3 => 4), (1:2 => 3:4, 1 => 2), (1:2 .=> 3:4, 1 => 2), missing)
    ranges = (Inf, 1)  # no missing here, because hopping range default is 1.0
    for r in rs, s in ss, dn in dns, rn in ranges
        model0 = hopping(1, region = r, sublats = s, dn = dn, range = rn, indices = inds) + onsite(1)
        model1 = hopping(1, region = r, sublats = s, dn = dn) + onsite(1)
        model2 = hopping(1, region = r, sublats = s) + onsite(1)
        model3 = hopping(1, region = r, range = rn) + onsite(1)
        model4 = hopping(1) + onsite(1)
        @test hopping(model1, region = r, sublats = s, dn = dn, range = rn, indices = inds) + onsite(model1) === model0
        @test hopping(model2, region = r, sublats = s, dn = dn, range = rn, indices = inds) + onsite(model2) === model0
        @test hopping(model3, region = r, sublats = s, dn = dn, range = rn, indices = inds) + onsite(model3) === model0
        @test hopping(model4, region = r, sublats = s, dn = dn, range = rn, indices = inds) + onsite(model4) === model0
    end
end

@testset "hopping adjoint" begin
    ts  = (1, 1im, @SMatrix[0 -im; im 0], @SMatrix[0 -im])
    ss  = (:A => :B, :A => (:A,:B), (:A,:B) .=> (:A,:B), (:A,:B) => (:B,:A), missing)
    ss´ = (:B => :A, (:A,:B) => :A, (:A,:B) .=> (:A,:B), (:B,:A) => (:A,:B), missing)
    dns = ((0,1), ((0,1),(1,0)),    SVector(0,1),  (SVector(0,1), (0,3)),    [1, 2], ([1.0,2], (0,4.0)),   missing)
    dns´= ((0,-1), ((0,-1),(-1,0)), SVector(0,-1), (SVector(0,-1), (0,-3)), -[1, 2], (-[1.0,2], (0,-4.0)), missing)
    ranges = (Inf, 1)
    for (t,t´) in zip(ts, adjoint.(ts)), (s, s´) in zip(ss, ss´), (dn, dn´) in zip(dns, dns´), rn in ranges
        hop = hopping(t, sublats = s, dn = dn, range = rn)
        hop´ = hopping(t´, sublats = s´, dn = dn´, range = rn)
        @test hop' === hop´
    end
end