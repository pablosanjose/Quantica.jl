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