using Quantica: nsubbands, nvertices, nedges, nsimplices
using LinearAlgebra: eigvals, Diagonal
using Random

@testset "extensions not loaded" begin
    solver = ES.ShiftInvert(ES.ArnoldiMethod(), 0.0)
    h = HP.graphene() |> supercell(10)
    @test_throws ArgumentError spectrum(h, (0,0); solver)
    solver = ES.Arpack(nev = 2)
    @test_throws ArgumentError spectrum(h, (0,0); solver)
end

using LinearMaps, ArnoldiMethod, KrylovKit, Arpack

@testset "basic bandstructures" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1))
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi, pi, 13), showprogress = false)
    @test nsubbands(b)  == 1

    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> supercell(2)
    b = bands(h, 0.99 * subdiv(-pi, pi, 13), 0.99 * subdiv(-pi, pi, 13), showprogress = false)
    @test nsubbands(b)  == 3

    h = LatticePresets.honeycomb() |>
        hamiltonian(onsite(0.5, sublats = :A) + onsite(-0.5, sublats = :B) +
                    hopping(-1, range = 1/√3))
    b = h |> bands(subdiv(-pi, pi, 13), subdiv(-pi, pi, 15), showprogress = false)
    @test nsubbands(b)  == 2

    h = LatticePresets.cubic() |> hamiltonian(hopping((r,dr)-> im*dr'SA[1,1.5,2])) |> supercell(2)
    b = bands(h, subdiv(-pi, pi, 5), subdiv(-pi, pi, 7), subdiv(-pi, pi, 5), showprogress = false)
    @test nsubbands(b)  == 1

    b = bands(h, subdiv(0, 1, 4), mapping = (:Γ, :X), showprogress = false)
    @test nsubbands(b)  == 1

    b = bands(h, subdiv(0, 4, 13), mapping = (:Γ, :X, (0, π), :Z, :Γ), showprogress = false)
    @test nsubbands(b)  == 1
    @test nvertices(b) == 32

    b = bands(h, subdiv(1:5, (4,5,6,7)), mapping = [1,2,3,5,4] => (:Γ, :X, (0, π), :Z, :Γ), showprogress = false)
    @test nsubbands(b) == 1
    @test nvertices(b) == 42

    b = bands(h, subdiv((1,3,4), 5), mapping = (1,4,3) => (:Γ, :X, :Z), showprogress = false)
    @test nsubbands(b) == 1
    @test nvertices(b) == 23

    # complex spectra
    h = LatticePresets.honeycomb() |> onsite(im) + hopping(-1) |> supercell(2)
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi,pi, 13), showprogress = false)
    @test nsubbands(b)  == 1
    # PR #419: non-normal spectra (nonreciprocal hopping), through the non-Hermitian eigen workspace of b's solver
    h = LatticePresets.honeycomb() |> onsite(im) + hopping((r, dr) -> -1 + 0.3*dr[1]) |> supercell(2)
    b = bands(h, subdiv(-pi, pi, 13), subdiv(-pi,pi, 13), showprogress = false)
    for φs in (SA[0.2, -0.7], SA[1.3, 0.4])
        sp = spectrum(b, φs)
        ε, ψ, H = energies(sp), states(sp), Matrix(h(φs))
        @test sort(ε, by = z -> (real(z), imag(z))) ≈ sort(eigvals(H), by = z -> (real(z), imag(z)))
        @test H * ψ ≈ ψ * Diagonal(ε)
    end

    # spectrum sorting
    h = LatticePresets.honeycomb() |> hopping(-1) |> supercell(10)
    b = bands(h, subdiv(0, 1, 7); solver = ES.Arpack(sigma = 0.1im, nev = 12), mapping = (:K, :K´), showprogress = false)
    @test nsubbands(b) == 1

    # defect insertion and patching
    h = LatticePresets.honeycomb() |> hopping(-1)
    b = bands(h, subdiv(0, 2pi, 8), subdiv(0, 2pi, 8); showprogress = false, defects = ((2pi/3, 4pi/3), (4pi/3, 2pi/3)), patches = 10)
    @test nvertices(b) == 140

    # knit min_squared_overlap should be < 0.5
    h = LP.square() |> hamiltonian(onsite(3*I) -hopping(I) + hopping((r, dr) -> 0.1*(im*dr[1]*SA[0 -im; im 0] - im*dr[2]*SA[0 1; 1 0])), orbitals = 2)
    b = bands(h, subdiv(-π, π, 19), subdiv(-π, π, 19))
    @test nsimplices(b) == 1312

    # PR #419: bands of a degenerate system reuse eigen and svd workspaces and the knitting colproj buffer
    # (~1.4-1.7 MiB, vs ~2.5-3 MiB if the svd workspace is resized to each new shape instead of
    # only grown, ~4 MiB without the eigen workspaces and ~8.5 MiB without any of them)
    h = LP.square() |> hamiltonian(hopping(I), orbitals = 2) |> supercell(2)
    let h = h, ks = subdiv(-π, π, 13)  # locals, to avoid global-scope allocations
        bands(h, ks, ks; showprogress = false)                                    # first call, to compile
        @test (@allocated bands(h, ks, ks; showprogress = false)) < 2 * 2^20      # 2 MiB
        # PR #419: unboxed captured variables in subbands_split! (~11k-18k allocs, vs ~24k-38k boxed)
        @test (@allocations bands(h, ks, ks; showprogress = false)) < 21000
    end
    # PR #419: simplex building reuses its buffers across vertices (~6-7 allocs, vs ~3.6k-9k)
    b = bands(HP.graphene(), subdiv(0, 2π, 30), subdiv(0, 2π, 30); showprogress = false)
    let neighs = Quantica.neighbors(Quantica.subbands(b)[1])                 # 1800 vertices
        Quantica.build_cliques(neighs, Val(3))                               # first call, to compile
        @test (@allocations Quantica.build_cliques(neighs, Val(3))) < 50
    end
    # PR #419: subband interval trees reuse one temporary list (~690-700 KiB, vs ~850-860 KiB)
    let sb = Quantica.subbands(b)[1], verts = Quantica.vertices(sb), simps = Quantica.simplices(sb)
        Quantica.subband_trees(verts, simps)                                 # first call, to compile
        @test (@allocated Quantica.subband_trees(verts, simps)) < 775 * 2^10 # 775 KiB
    end
end

@testset "functional bandstructures" begin
    hc = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, sublats = :A=>:B) |> plusadjoint) |> supercell(3)
    hf1((x,)) = Matrix(Quantica.call!(hc, (x, -x)))
    m = subdiv(0, 1, 4)
    b = bands(hf1, m, showprogress = false, mapping = x -> 2π * x)
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 36
    # teting thread safety - we should fall back to a single thread for hf2::Function
    hf2((x,)) = Quantica.call!(hc, (x, -x))
    m = subdiv(0,2π,40)
    Random.seed!(1) # to have ArnoldiMethod be deterministic
    b = bands(hf2, m, showprogress = false, solver = ES.ArnoldiMethod(nev = 18))
    # not a good test: there is a random, platform-dependent component to this, so it sometimes fails
    @test_skip nsubbands(b) <= 2

    hp2 = LatticePresets.honeycomb() |> hamiltonian(hopping(-1), @hopping!((t; s) -> s*t))
    hf3((s, x)) = Matrix(Quantica.call!(hp2, (x, x); s))
    m = subdiv(0, pi, 13)
    b = bands(hf3, m, m, showprogress = false)
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 288 * 2
    b = bands(hf3, m, m, showprogress = false, transform = ω -> ω^2)    # degeneracies doubled
    @test nsubbands(b) == 1
    @test nsimplices(b)  == 288
end

@testset "parametric bandstructures" begin
    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> supercell(2) |>
         hamiltonian(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    mesh2D = subdiv(0, 1, 15), subdiv(0, 2π, 15)
    b = bands(ph, mesh2D..., mapping = (x, k) -> ftuple(x; k = k), showprogress = false)
    @test nsubbands(b)  == 4
    b = bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(1; k = k, p = SA[1, φ]), showprogress = false)
    @test nsubbands(b)  == 1

    ph = LatticePresets.linear() |> hamiltonian(onsite(0I) + hopping(-I), orbitals = Val(2)) |> supercell(2) |>
        supercell |> hamiltonian(@onsite!((o; k) -> o + k*I), @hopping!((t; k = 2, p = [1,2])-> t - k*I .+ p'p))
    b = bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(; k = k, p = SA[1, φ]), showprogress = false)
    @test nsubbands(b)  == 1
    # multithreading loop does not throw error
    b = bands(ph, mesh2D..., mapping = (k, φ) -> ftuple(; k, p = SA[1, φ]), showprogress = false)
    @test nsubbands(b) == 1
end

@testset "spectrum" begin
    h = LatticePresets.honeycomb() |> hamiltonian(onsite(2I) + hopping(I, range = 1), orbitals = (Val(2), Val(1))) |> supercell(2) |> supercell
    for solver in (ES.LinearAlgebra(), ES.Arpack(), ES.KrylovKit(), ES.ArnoldiMethod(), ES.ShiftInvert(ES.ArnoldiMethod(), 0.4))
        sp = spectrum(h; solver)
        e1, s1 = sp
        e2, s2 = first(sp), last(sp)
        e3, s3 = energies(sp), states(sp)
        @test e1 === e2 === e3
        @test s1 === s2 === s3
        @test Tuple(sp) === (e1, s1)
        for sp´ in (sp[[3,5]], sp[1:2], sp[[2,3], around = 0])
            @test length(sp´[1]) == 2
            @test size(sp´[2]) == (Quantica.flatsize(h), 2)
        end
        @test sp[around = 0] isa Tuple
        @test sp[2, around = 0] isa Tuple
        @test sp[[2,3,4], around = 0] == sp[2:4, around = 0]
        @test sp[[2,3,4], around = 0] !== sp[2:4, around = 0]
        @test sp[[2,4,3], around = -Inf][2] == hcat(sp[2][2], sp[4][2], sp[3][2])
    end
end

@testset "bandstructures/spectrum slices" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, range = 1/√3)) |> supercell(4)
    b = bands(h, subdiv(0, 2pi, 13), subdiv(0, 2pi, 15), showprogress = false)
    @test b[(:, pi)] isa Vector{Quantica.Subband{Float64,2,Missing}}
    @test length(b[(pi,)]) == length(b[(pi, :)]) == length(b[(pi, :, :)]) == 1
    @test b[1] isa Quantica.Subband
    @test b[[1,end]] isa Vector{<:Quantica.Subband}
    m = Quantica.slice(b, (:,1))
    @test only(m) isa Quantica.Mesh
    m = Quantica.slice(b[1], (:,1))
    @test m isa Quantica.Mesh

    s = spectrum(b, (pi, pi))
    s´ = spectrum(stitch(h, (pi, pi)))
    s´´ = spectrum(h, (pi, pi))
    ϵs, ψs = ES.LinearAlgebra()(Matrix(h((pi, pi))))
    @test s isa Quantica.Spectrum
    @test Tuple(s) == Tuple(s´) == Tuple(s´´) == (ϵs, ψs)
    @test s[1:10, around = 0] == s´[1:10, around = 0] == s´´[1:10, around = 0]
end

@testset "gap in 1D bands" begin
    h = LP.linear() |> supercell(2) |> hopping(1) - @onsite((r; U = 0) ->ifelse(iseven(r[1]), U, -U))
    @test Quantica.gap(h(U = 1)) ≈ 1
    @test Quantica.gap(h(U = 0.1)) ≈ 0.1
    @test Quantica.gap(h(U = 0.0)) ≈ 0.0
end

@testset "berry curvature" begin
    SOC(dr) = ifelse(iseven(round(Int, atan(dr[2], dr[1])/(pi/3))), im, -im)
    model = hopping(1) + @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :A => :A, range = 1) - @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :B => :B, range = 1)
    h = LatticePresets.honeycomb(a0 = 1) |> model
    bc = berry_curvature(h(α = 0.05))
    chern = mean([bc(SA[ϕ1,ϕ2],1) for ϕ1 in range(0, 2pi, 101)[1:end-1], ϕ2 in range(0, 2pi, 101)[1:end-1]])/2π
    @test chern ≈ 1
    bc = berry_curvature(h)
    chern = mean([bc(SA[ϕ1,ϕ2],2; α = 0.05) for ϕ1 in range(0, 2pi, 101)[1:end-1], ϕ2 in range(0, 2pi, 101)[1:end-1]])/2π
    @test chern ≈ -1
    # Non-Abelian case
    h2 = h |> supercell(2)
    bc = berry_curvature(h2, maxdim = 3)
    @test length(bc(SA[2π/3, -2π/3]; α = 0.0)) == 3
    @test isapprox(bc(SA[2π/3, -2π/3], 4:5; α = 0.0), [0 0; 0 0]; atol = 1e-8)
end
