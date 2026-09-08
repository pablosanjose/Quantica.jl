############################################################################################
# Scattering
#   Represents a scattering problem with a central 0D system and N contacts, all of which
#   should be GreenFunctionSchurEmptyLead1D.
# ScatteringSolution
#   A scattering problem solved for specific energy ω and params. It contains all building
#   blocks required to compute incoming and outgoing scattering states, but they are not
#   built explicitly until calling scatteringstate. It does contain all the blocks of the
#   scattering matrix, which can be access via sω[j,i]
# ScatteringState
#   Built with scatteringstate(sω::ScatteringSolution, lead::Int; modes = 1, cells = 1)
#   Can be passed to qplot(ss::ScatteringState; kw...) where shaders are functions of the
#   wavefunction at a given point.
#region

struct ScatteringWorkspace{G<:GreenFunction}
    g::G
    Σ::Vector{Matrix{ComplexF64}}
    G::Matrix{ComplexF64}
    S::Matrix{ComplexF64}
end

struct Scattering{G<:GreenFunction}
    g::G
    workspace::ScatteringWorkspace{G}
end

struct ScatteringSolution{G<:GreenSolution}
    gω::G
end

## API ##

function scattering(g)
    check_contacts_are_schur(g)
    workspace = ScatteringWorkspace(g)
    return Scattering(g, workspace)
end

function check_contacts_are_schur(g)
    foreach(contacts(g)) do c
        solver(c) isa GreenFunctionSchurEmptyLead1D || argerror("All contacts must be GreenFunctionSchurEmptyLead1D")
    end
    return g
end

#endregion
