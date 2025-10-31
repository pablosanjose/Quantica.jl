@test_throws ArgumentError qplot(LP.linear())  # no backend loaded

using CairoMakie

@testset "plot lattice" begin
    # Makie issue #3009 workaround
    lat = LP.linear() |> supercell(10) |> supercell
    @test display(qplot(lat)) isa CairoMakie.Screen{CairoMakie.IMAGE}
end

@testset "plot hamiltonian" begin
    h = LP.bcc() |> hamiltonian(hopping(1)) |> supercell(3) |> supercell((1,0,0))
    g = greenfunction(h)
    @test qplot(h, sitecolor = :blue, siteradius = 0.3, inspector = true) isa Figure
    @test qplot(h, sitecolor = :yellow, siteopacity = (i, r) -> r[1], inspector = true, flat = false) isa Figure
    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> supercell(3) |> supercell
    g = h |> attach(nothing) |> greenfunction
    @test qplot(h, hopcolor = :blue, hopradius = ldos(g(0.2)), inspector = true) isa Figure
    @test qplot(h, hopcolor = (:blue, RGBAf(1,0,0)), sitecolor = [:orange, :white], inspector = true) isa Figure
    @test qplot(h, hopcolor = :yellow, hopopacity = current(g(0.2)), inspector = true, flat = false) isa Figure
    @test qplot(g, hopcolor = :yellow, hopopacity = (ij, (r, dr)) -> r[1], inspector = true, flat = false) isa Figure
    @test scatter(h, :A) isa Makie.FigureAxisPlot
    @test scatter(g, 1) isa Makie.FigureAxisPlot
    @test scatter(lattice(g)) isa Makie.FigureAxisPlot
    @test lines(h, :A) isa Makie.FigureAxisPlot
    @test lines(g) isa Makie.FigureAxisPlot
    @test_throws BoundsError lines(lattice(h), 3)
    h = LP.linear() |> hamiltonian(@hopping(()->I), orbitals = 2)
    @test qplot(h) isa Figure
    g = LP.linear() |> hamiltonian(hopping(I), orbitals = 2) |> attach(@onsite(ω->im*I), cells = 1) |> attach(@onsite(ω->im*I), cells = 4) |> greenfunction
    @test qplot(g, selector = siteselector(; cells = -10:10), children = (; sitecolor = :blue)) isa Figure
    # matrix shader
    gx1 = abs2.(g(0.01)[siteselector(cells = 1:10), 1])
    @test qplot(g, selector = siteselector(cells = 1:10), sitecolor = gx1) isa Figure
    # vector shader
    gx1´ = vec(sum(gx1, dims = 2))
    @test qplot(g, selector = siteselector(cells = 1:10), sitecolor = gx1´) isa Figure
    # green with leads
    glead = LP.honeycomb() |> hopping(1, range = 1) |> supercell((1,-1), region = r -> 0<=r[2]<=5) |> attach(nothing, cells = SA[5]) |> greenfunction(GS.Schur(boundary = 0));
    g = LP.honeycomb() |> hopping(1) |> supercell(region = r -> -6<=r[1]<=6 && 0<=r[2]<=5) |> attach(glead, region = r -> r[1] > 4.9) |> greenfunction;
    @test qplot(g, shellopacity = 0.3) isa Figure
    hlead = LP.square() |> supercell((1,0), region = r -> 0 <= r[2] < 2) |> hopping(1)
    glead = LP.square() |> onsite(4) - hopping(1) |> supercell((1, 0), region = r -> abs(r[2]) <= 5/2) |> attach(nothing, cells = SA[2]) |> greenfunction(GS.Schur(boundary = -2))
    @test qplot(glead, siteradius = 0.25, children = (; sitecolor = :blue)) isa Figure
    g = LP.honeycomb() |> hopping(1, range = 1) |>
        attach(nothing, region = RP.circle(1, SA[2,3])) |> attach(nothing, region = RP.circle(1, SA[3,-3])) |>
        greenfunction(GS.Bands(subdiv(-π, π, 13), subdiv(-π, π, 13), boundary = 2=>-3))
    @test qplot(g) isa Figure
    # Issue 200
    g = LP.linear() |> hamiltonian(@hopping((; q = 1) -> q*I), orbitals = 2) |> attach(@onsite((ω; p = 0) ->p*SA[0 1; 1 0]), cells = 1) |> greenfunction
    @test qplot(g) isa Figure
    @test qplot(g(p = 3)) isa Figure
    # Issue 243
    oh = LP.linear() |> hopping(1) |> attach(@onsite((ω; p = 1) -> p), cells = 1) |> attach(@onsite((ω; p = 1) -> p), cells = 3)
    @test qplot(oh) isa Figure
end

@testset "plot bands" begin
    SOC(dr) = ifelse(iseven(round(Int, atan(dr[2], dr[1])/(pi/3))), im, -im)
    model = hopping(1) + @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :A => :A, range = 1) - @hopping((r, dr; α = 0) -> α * SOC(dr); sublats = :B => :B, range = 1)
    h = LatticePresets.honeycomb(a0 = 1) |> hamiltonian(model)
    b = bands(h(α = 0.05), range(0, 2pi, length=60), range(0, 2pi, length = 60))
    @test qplot(b, color = (psi, e, k) -> angle(psi[1] / psi[2]), colormap = :cyclic_mrybm_35_75_c68_n256, inspector = true) isa Figure
    cs = Makie.ColorScheme([colorant"red", colorant"black"])
    @test qplot(b, color = (psi, e, k) -> angle(psi[1] / psi[2]), colormap = cs, inspector = true) isa Figure
    cs = Makie.ColorScheme([colorant"red", colorant"white", colorant"blue"])
    b = bands(h(α = 0.05), range(0, 2pi, length=60), range(0, 2pi, length = 60); metadata = berry_curvature(h(α = 0.05)))
    @test qplot(b, color = (psi, e, k, m) -> m, colormap = cs, inspector = true) isa Figure
end
