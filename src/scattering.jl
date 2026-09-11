############################################################################################
# Scattering
#   Represents a scattering problem with a central 0D system and N contacts
# ScatteringSolution
#   A scattering problem solved for specific energy ω and params. It contains all building
#   blocks required to compute incoming and outgoing scattering states, but they are not
#   built explicitly until calling scatteringstates. It does contain all the blocks of the
#   scattering matrix, which can be access via sω[j,i]
# scatteringstates
#   Built with scatteringstates(sω::ScatteringSolution, lead::Int; modes = 1, cells = 1)
#   Can be passed to qplot(ss::scatteringstates; kw...) where shaders are functions of the
#   wavefunction at a given point.
#region

struct LeadSolution{T}
    phiR::Matrix{Complex{T}}     # reflected wave φʳR = (iG₁₁Γ-1)Φₐ where G₁₁=gʳ+gʳV'GVgʳ
    gh::Matrix{Complex{T}}       # Transfer matrix gʳh₊
    phi_a::Matrix{Complex{T}}    # incoming modes Φₐ
    lambda_a::Vector{Complex{T}} # eigenvalues of incoming modes λₐ, so Λₐ = Diagonal(λₐ)
    source::Matrix{Complex{T}}   # source in central region V'(Φₐ - gʳh₊ΦₐΛₐ⁻¹)
end

struct ScatteringWorkspace{T}
    ll::Matrix{Complex{T}}       # lead-lead intermediate preallocation
    ll´::Matrix{Complex{T}}       # lead-lead intermediate preallocation
    lc::Matrix{Complex{T}}       # lead-central intermediate preallocation
    cl::Matrix{Complex{T}}       # central-lead intermediate preallocation
    leadsol::LeadSolution{T}     # solution preallocation
end

struct Scattering{T,N,G<:GreenFunction{T},W<:NTuple{N,Union{Nothing,ScatteringWorkspace{T}}}}
    g::G
    workspaces::W                # one workspace per contact (nothing for non-Schur leads)
end

struct ScatteringSolution{T,N,G<:GreenSolution{T},S<:NTuple{N,Union{Nothing,LeadSolution{T}}}}
    gω::G
    leadsols::S                  # any lead that is not empty Schur has `nothing` lead solution
end

struct scatteringstates{T,H<:Hamiltonian{T}}
    h_with_leads::H              # combined Hamiltonian with a number of lead unit cells
    state::Matrix{Complex{T}}    # columns are scattering states for a given incoming mode
end

## Constructors ##

function scattering(g)
    workspaces = ScatteringWorkspace.(solver.(selfenergies(g)))
    all(isnothing, workspaces) && argerror("At least one contact must be a Schur lead without additional self-energies")
    return Scattering(g, workspaces)
end

# fallback
ScatteringWorkspace(_) = nothing

function ScatteringWorkspace(s::Union{SelfEnergySchurSolver{T},SelfEnergyCouplingSchurSolver{T}}) where {T}
    (nc, nl) = size(coupling_from_lead(s))
    d = deflated_dimension(s)
    ll = Matrix{Complex{T}}(undef, nl, nl)
    ll´ = similar(ll)
    lc = Matrix{Complex{T}}(undef, nl, nc)
    cl = Matrix{Complex{T}}(undef, nc, nl)
    leadsol = LeadSolution{T}(nl, nc, d)
    return ScatteringWorkspace(ll, ll´, lc, cl, leadsol)
end

function LeadSolution{T}(nl, nc, d) where {T}
    phiR = Matrix{Complex{T}}(undef, nl, nl)
    gh = Matrix{Complex{T}}(undef, nl, nl)
    phi_a = Matrix{Complex{T}}(undef, nl, d)
    lambda_a = Vector{Complex{T}}(undef, d)
    source = Matrix{Complex{T}}(undef, nc, d)
    return LeadSolution(phiR, gh, phi_a, lambda_a, source)
end


## API ##

function call!(s::Scattering{<:Any,N}, ω; params...) where {N}
    # This invokes any SchurFactorSolver with this ω and params,
    # so its Schur factors and modes are populated after this point
    Gω = call!(s.g, ω; params..., skipmodes_internal = false)
    solvers = solver.(selfenergies(s.g))
    leadinds = ntuple(identity, Val(N))
    leadsols = solve_lead.(solvers, Ref(Gω), leadinds, s.workspaces)
    return ScatteringSolution(Gω, leadsols)
end

(s::Scattering)(ω; params...) = copy(call!(s, ω; params...))

Base.copy(s::ScatteringSolution) = ScatteringSolution(s.gω, copy.(s.leadsols))
Base.copy(s::LeadSolution) =
    LeadSolution(copy(s.phiR), copy(s.gh), copy(s.phi_a), copy(s.lambda_a), copy(s.source))

## SelfEnergySchurSolver lead solution ##

# This is the uniform coupling case, for which φʳR = gʳh₊(iG₀₀Γ-1)Φₐ
# The source term reads source = iΓΦₐΛₐ⁻¹
function solve_lead(solver::SelfEnergySchurSolver, Gω, leadindex, sw::ScatteringWorkspace)
    leadsol, Γ, ll = sw.leadsol, sw.ll´, sw.ll
    G00 = Gω[leadindex, leadindex]
    hm, _ = couplings_intralead(solver)
    grhp = outgoing_gh(solver)
    λa, Φa = incoming_λΦ(solver)

    # Building Γ = i(h₋gʳh₊ - (h₋gʳh₊)')
    mul!(Γ, hm, grhp, im, 0)        # ih₋gʳh₊
    ll .= Γ'
    Γ .+= ll                        # ih₋gʳh₊ - i(h₋gʳh₊)')

    # Populating lead solution
    @show size(leadsol.phi_a), size(Φa)
    copy!(leadsol.phi_a, Φa)
    copy!(leadsol.lambda_a, λa)
    copy!(leadsol.gh, grhp)
    mul!(leadsol.source, Γ, Φa, im, 0)
    leadsol.source ./= transpose(λa)    # iΓΦₐΛₐ⁻¹

    # Building φʳR = (iG₀₀Γ-1)Φₐ
    copyto!(ll, -I)
    mul!(ll, G00, Γ, im, 1)         # (iG₀₀Γ-1)
    mul!(leadsol.phiR, grhp, mul!(Γ, ll, Φa)) # φʳR = gʳh₊(iG₀₀Γ-1)Φₐ, we reuse Γ as temporary

    return leadsol
end

#endregion
