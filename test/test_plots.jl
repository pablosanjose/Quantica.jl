using CairoMakie

@testset "plot hamiltonian" begin
    h = LP.bcc() |> hamiltonian(hopping(1)) |> supercell(3) |> supercell((1,0,0))
    g = greenfunction(h)
    @test qplot(h, sitecolor = :blue, siteradius = 0.3, inspector = true) isa Figure
    @test qplot(h, sitecolor = :yellow, siteopacity = (i, r) -> r[1], inspector = true, flat = false) isa Figure
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> supercell(3) |> supercell
    g = h |> attach(nothing) |> greenfunction
    @test qplot(h, hopcolor = :blue, hopradius = ldos(g(0.2)), inspector = true) isa Figure
    @test qplot(h, hopcolor = :yellow, hopopacity = current(g(0.2)), inspector = true, flat = false) isa Figure
    @test qplot(g, hopcolor = :yellow, hopopacity = (ij, (r, dr)) -> r[1], inspector = true, flat = false) isa Figure
    @test scatter(h, :A) isa Makie.FigureAxisPlot
    @test scatter(g, 1) isa Makie.FigureAxisPlot
    @test scatter(lattice(g)) isa Makie.FigureAxisPlot
    @test lines(h, :A) isa Makie.FigureAxisPlot
    @test lines(g) isa Makie.FigureAxisPlot
    @test_throws BoundsError lines(lattice(h), 3)
    h = LP.linear() |> hamiltonian(@hopping(()->I), orbitals = 2)
    @test qplot(g) isa Figure
    g = LP.linear() |> hamiltonian(hopping(I), orbitals = 2) |> attach(@onsite(ω->im*I), cells = 1) |> attach(@onsite(ω->im*I), cells = 4) |> greenfunction
    @test qplot(g, cells = -10:10) isa Figure
end

@testset "plot bands" begin
    SOC(dr) = ifelse(iseven(round(Int, atan(dr[2], dr[1])/(pi/3))), im, -im)
    model = hopping(1, range = 1/√3) + @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :A => :A, range = 1) - @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :B => :B, range = 1)
    h = LatticePresets.honeycomb(a0 = 1) |> hamiltonian(model)
    b = bands(h(α = 0.05), range(0, 2pi, length=60), range(0, 2pi, length = 60))
    @test qplot(b, color = (psi, e, k) -> angle(psi[1] / psi[2]), colormap = :cyclic_mrybm_35_75_c68_n256, inspector = true) isa Figure
end