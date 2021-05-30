@testset "single-column kets" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(3)))
    k = ket(h)
        @test iszero(k)
        @test eltype(k) <: SVector{3}
        @test size(k) == (size(h, 1), 1)
    @test_throws ArgumentError ket(ketmodel(1), h)
    @test_throws ArgumentError ket(ketmodel(r -> SA[1,2,3]), h)
    @test_throws ArgumentError ket(ketmodel(2, sublats = :B), h)
    @test_throws ArgumentError ket(ketmodel(SA[1,2,3], maporbitals = true), h)
    k = ket(ketmodel(SA[1], maporbitals = true), h)
        @test parent(k) == SVector.(hcat([(0.5,0,0); (0.5,0.5,0.5)]))
    k = ket(ketmodel(fill(1, 1, 1), maporbitals = true), h)
        @test parent(k) == SVector.(hcat([(0.5,0,0); (0.5,0.5,0.5)]))
    k = ket(ketmodel(r -> [1], maporbitals = true), h)
        @test parent(k) == SVector.(hcat([(0.5,0,0); (0.5,0.5,0.5)]))
    k = ket(ketmodel(r -> SA[3], maporbitals = true, sublats = :A), h)
        @test parent(k) == SVector.(hcat([(1,0,0); (0,0,0)]))
    k = ket(ketmodel(2, sublats = :A), h)
        @test parent(k) == SVector.(hcat([(1,0,0); (0,0,0)]))
    k = ket(ketmodel(SA[1, 2, 2], sublats = :B), h)
        @test parent(k) == SVector.(hcat([(0,0,0); (1/3,2/3,2/3)]))
    k = ket(ketmodel(r -> SMatrix{3,1}((1, 2, 2)), sublats = :B), h)
        @test parent(k) == SVector.(hcat([(0,0,0); (1/3,2/3,2/3)]))
    k = ket(ketmodel(r -> 2r[2], sublats = :A), h)
        @test parent(k) == SVector.(hcat([(-1,0,0); (0,0,0)]))
    k = ket(ketmodel(2, sublats = :A, normalization = missing), h)
        @test parent(k) == SVector.(hcat([(2,0,0); (0,0,0)]))
    k = ket(ketmodel(1; maporbitals = true), h)
        @test parent(k) == SVector.(hcat([ (0.5,0,0); (0.5,0.5,0.5)]))
    k = ket(ketmodel(1; maporbitals = true), h)
        @test parent(k) == SVector.(hcat([(0.5,0,0); (0.5,0.5,0.5)]))
end

@testset "multi-column kets" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(3)))
    k = ket(ketmodel(SA[1 1]; maporbitals = true), h)
        @test parent(k) == SVector.([(0.5,0,0) (0.5,0,0); (0.5,0.5,0.5) (0.5,0.5,0.5)])
    k´ = ket(ketmodel([1 1]; maporbitals = true), h)
        @test k == k´
    @test_throws ArgumentError ket(ketmodel(SA[1 0; 0 1]; maporbitals = true), h)
    @test_throws ArgumentError ket(ketmodel([1 0; 0 2; 0 1]; sublats = :A), h)
    @test_throws ArgumentError ket(ketmodel(SA[1 0; 0 1]), h)
    @test_throws ArgumentError ket(ketmodel(SA[1 0 0; 0 2 0; 0 0 3]), h)
    k = ket(ketmodel(SA[1 0; 0 2; 0 1], sublats = :B, normalization = missing), h)
        @test parent(k) == SVector.([(0,0,0) (0,0,0); (1,0,0) (0,2,1)])
    k´ = ket(ketmodel([1 0 0; 0 2 1]', sublats = :B, normalization = missing), h)
    k´´= ket(ketmodel(r -> SA[1 0 0; 0 2 1]', sublats = :B, normalization = missing), h)
        @test k == k´´
    @test_throws Exception ket(ketmodel(2I), h)
    k = ket(ketmodel([1 0; 0 2; 0 1], sublats = :B, normalization = missing), h)
        @test parent(k) == SVector.([(0,0,0) (0,0,0); (1,0,0) (0,2,1)])
end

@testset "randomkets" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = Val(2))
    ks = ket.(randomkets(3, r-> 2*SA[1 0; 0 1], normalization = missing), Ref(h))
        @test length(ks) == 3
        @test ks isa Vector{<:Quantica.Ket}
        @test ks[1] == ks[2] == ks[3]
    k = first(ks)
        @test parent(k) == SVector.([(2,0) (0,2); (2,0) (0,2)])
    k1, k2 = ket.(randomkets(2, r-> randn()*SA[1 0; 0 1], normalization = missing), Ref(h))
        r, s = k1[1,1][1], k1[2,1][1]
        @test k1 ≈ SVector.([(r,0) (0,r); (s,0) (0,s)])
        r, s = k2[1,1][1], k2[2,1][1]
        @test k2 ≈ SVector.([(r,0) (0,r); (s,0) (0,s)])
end

@testset "flatten/unflatten kets" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = (Val(1), Val(3)))
    k = ket(ketmodel(SA[1 1]; maporbitals = true), h)
    @test unflatten(flatten(k), orbitalstructure(k)) == k
end