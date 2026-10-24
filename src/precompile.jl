############################################################################################
# Precompile workload
#   Runs a representative set of calls at package precompile time, so that their compiled
#   code is cached in the package image. Covers common lattice/hamiltonian construction,
#   all GreenSolvers, the usual slicing patterns and the main observables, for Float64
#   with scalar, 2x2 and mixed (2,1) orbitals. Code involving user closures (e.g.
#   @hopping((r, dr) -> ...)) cannot be precompiled this way, as each closure is a new type.
#region

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    ω = 0.2
    @compile_workload begin
        # lattices and hamiltonians, scalar, 2x2 and mixed orbitals
        for orbs in (1, 2, (2, 1))
            h2D = LP.honeycomb() |> hamiltonian(onsite(I) - hopping(I), orbitals = orbs)
            h2D((0.1, 0.2))
            hdot = h2D |> supercell(region = RP.circle(2))
            hdot[()]
            hlead = h2D |> supercell((1, -1), region = r -> abs(r[2]) < 2)
            hlead((0.3,))

            # GreenSolvers and slicing
            for g in (greenfunction(hdot, GS.SparseLU()), greenfunction(hdot, GS.Spectrum()))
                gω = g(ω)
                gω[cells = ()]
                gω[sites(1:2), sites(1)]
                gω[diagonal(cells = ())]
                ldos(gω)
            end
            glead = greenfunction(hlead, GS.Schur(boundary = 0))
            glead(ω)[cells = 1]
            glead(ω)[sitepairs(range = 1)]

            # contacts: nothing, model self-energy and lead
            g = hdot |> attach(nothing, region = RP.circle(1)) |>
                attach(@onsite(ω -> -im * I), region = r -> r[1] > 1.5) |>
                attach(glead, region = r -> r[1] < -1.5) |> greenfunction
            gω = g(ω)
            gω[1]
            gω[2, 3]
            gω[cells = ()]

            # observables
            conductance(g[2, 3])(ω)
            transmission(g[2, 3])(ω)
            ldos(g[1])(ω)
            current(g[cells = ()])(ω)
            densitymatrix(g[1], 3)()
            orbs == 2 && josephson(g[3], 2)()  # requires uniform, even orbitals (Nambu)
        end

        # other solvers, observables and spectra (scalar orbitals)
        h = LP.honeycomb() |> hopping(1)
        hsc = h |> supercell(region = RP.circle(3))
        densitymatrix(greenfunction(hsc, GS.Spectrum())[cells = ()])()
        gkpm = hsc |> attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 20, bandrange = (-3, 3), showprogress = false))
        ldos(gkpm[1])(ω)
        gbands = h |> attach(nothing, cells = (0, 0)) |>
            greenfunction(GS.Bands(subdiv(0, 2π, 5), subdiv(0, 2π, 5), showprogress = false))
        gbands(ω)[1]
        spectrum(h, (0.1, 0.2))
        bands(h, subdiv(0, 2π, 5), subdiv(0, 2π, 5), showprogress = false)
        hlead = h |> supercell((1, -1), region = r -> abs(r[2]) < 2)
        densitymatrix(greenfunction(hlead, GS.Schur())[cells = 0])()
    end
end

#endregion
