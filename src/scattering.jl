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

# PhiS: PhiSᵢᵢ = φʳRᵢᵢ = (iGᵢᵢΓ-1)Φₐᵢ and PhiSᵢⱼ = φʳTᵢⱼ = iGᵢⱼΓΦₐⱼ for i≠j
# S: Sᵢᵢ = rᵢᵢ = √VᵣᵢᵖPᵣᵢRᵢᵢPₐⱼ⁺/√(-Vᵣⱼᵖ) and Sᵢⱼ = tᵢⱼ = √VᵣᵢᵖPᵣᵢTᵢⱼPₐⱼ⁺/√(-Vₐⱼᵖ) for i≠j
# Here: Gᵢᵢ=gᵢʳ+gᵢʳV'GVgᵢʳ and Gᵢⱼ=gᵢʳV'GVgⱼʳ are intra/inter surface dressed propagators,
#       Γ = i(h₋gʳh₊ - (h₋gʳh₊)') are intra-lead decay rates
#       Pᵣᵢ are projector onto outgoing propagating modes in lead i, with velocity Vᵣᵢᵖ
#       Pₐⱼ are projector onto incoming propagating modes in lead j, with velocity Vₐⱼᵖ
struct ScatteringMatrix{T}
    PhiS::Matrix{Matrix{Complex{T}}}  # scattering information for all modes
    S::Matrix{Matrix{Complex{T}}}     # scattering matrix for propagating modes
end

struct ScatteringWorkspace{T}
    cl_ij::Matrix{Matrix{Complex{T}}} # central-lead intermediate preallocations foreach i,j
    lc_i::Vector{Matrix{Complex{T}}}  # lead-central intermediate preallocations foreach i
    smat::ScatteringMatrix{T}         # solution preallocation
end

struct LeadSolution{T}
    gh::Matrix{Complex{T}}            # Transfer matrix gʳh₊
    phi_a::Matrix{Complex{T}}         # incoming modes Φₐ
    lambda_a::Vector{Complex{T}}      # eigenvalues of incoming modes λₐ, so Λₐ = Diagonal(λₐ)
    ggpa::Matrix{Complex{T}}          # gʳΓΦₐ matrix, required to compute scattering matrix
    prpg::Matrix{Complex{T}}          # Φᵣ'Γ matrix, required to compute scattering matrix
    source::Matrix{Complex{T}}        # source in central region V'(Φₐ - gʳh₊ΦₐΛₐ⁻¹)
end

struct LeadWorkspace{T}
    ll::Matrix{Complex{T}}            # lead-lead intermediate preallocation
    leadsol::LeadSolution{T}          # solution preallocation
end

struct Scattering{T,N,G<:GreenFunction{T},W<:NTuple{N,Union{Nothing,LeadWorkspace{T}}}}
    g::G
    leadtmps::W                       # one workspace per contact (nothing for non-Schur leads)
    scattmp::ScatteringWorkspace{T}   # serves as preallocated storage for scattering matrix blocks
end

struct ScatteringSolution{T,N,G<:GreenSolution{T},S<:NTuple{N,Union{Nothing,LeadSolution{T}}}}
    gω::G
    leadsols::S                  # any lead that is not empty Schur has `nothing` lead solution
    matrix::ScatteringMatrix{T}  # and its corresponding smat blocks are empty matrices
end

struct ScatteringStates{T,H<:Hamiltonian{T}}
    h_with_leads::H              # combined Hamiltonian with a number of lead unit cells
    state::Matrix{Complex{T}}    # columns are scattering states for a given incoming mode
end

## Constructors ##

function scattering(g::GreenFunction{T}) where {T}
    leadtmps = LeadWorkspace.(solver.(selfenergies(g)))
    all(isnothing, leadtmps) &&
        argerror("At least one contact must be a Schur lead without additional self-energies")
    scattmp = ScatteringWorkspace(leadtmps)
    return Scattering(g, leadtmps, scattmp)
end

# fallback
LeadWorkspace(_) = nothing

function LeadWorkspace(s::Union{SelfEnergySchurSolver{T},SelfEnergyCouplingSchurSolver{T}}) where {T}
    (nl, nc) = size(first(coupling_to_from_lead(s)))
    d = deflated_dimension(s)
    ll = Matrix{Complex{T}}(undef, nl, nl)
    leadsol = LeadSolution{T}(nl, nc, d)
    return LeadWorkspace(ll, leadsol)
end

# at least one of the leadworkspace is not Nothing, otherwise this is never called
function ScatteringWorkspace(leadtmps::NTuple{<:Any,Union{Nothing,LeadWorkspace{T}}}) where {T}
    cl_ij = [Matrix{Complex{T}}(undef, central_flatsize(i), lead_flatsize(j)) for i in leadtmps, j in leadtmps]
    lc_i = [Matrix{Complex{T}}(undef, lead_flatsize(i), central_flatsize(i)) for i in leadtmps]
    smat = ScatteringMatrix(leadtmps)
    return ScatteringWorkspace(cl_ij, lc_i, smat)
end

function LeadSolution{T}(nl, nc, d) where {T}
    gh = Matrix{Complex{T}}(undef, nl, nl)
    phi_a = Matrix{Complex{T}}(undef, nl, d)
    lambda_a = Vector{Complex{T}}(undef, d)
    ggpa = Matrix{Complex{T}}(undef, nl, d)
    prpg = Matrix{Complex{T}}(undef, d, nl)
    source = Matrix{Complex{T}}(undef, nc, d)
    return LeadSolution(gh, phi_a, lambda_a, ggpa, prpg, source)
end

# at least one of the leadworkspace is not Nothing, otherwise this is never called
function ScatteringMatrix(leadtmp::NTuple{<:Any,Union{Nothing,LeadWorkspace{T}}}) where {T}
    N = length(leadtmp)
    S = Matrix{Matrix{Complex{T}}}(undef, N, N)
    PhiS = Matrix{Matrix{Complex{T}}}(undef, N, N)
    for j in 1:N, i in 1:N
        dimi = lead_flatsize(leadtmp[i])
        dimj = lead_flatsize(leadtmp[j])
        S[i,j] = Matrix{Complex{T}}(undef, dimi, dimj)
        PhiS[i,j] = Matrix{Complex{T}}(undef, dimi, dimj)
    end
    return ScatteringMatrix(PhiS, S)
end

lead_flatsize(::Nothing) = 0
lead_flatsize(lw::LeadWorkspace) = size(lw.ll, 1)

# computes the maximum of nc and nl for a given lead
central_flatsize(::Nothing) = 0
central_flatsize(lw::LeadWorkspace) = size(lw.leadsol.source, 1)


## API ##

function call!(s::Scattering{<:Any,N}, ω; params...) where {N}
    # This invokes any SchurFactorSolver with this ω and params,
    # so its Schur factors and modes are populated after this point
    Gω = call!(s.g, ω; params..., skipmodes_internal = false)
    solvers = solver.(selfenergies(s.g))
    leadsols = solve_lead.(s.leadtmps, solvers)
    smat = compute_scattering_matrix!(s, Gω, leadsols, solvers)
    return ScatteringSolution(Gω, leadsols, smat)
end

(s::Scattering)(ω; params...) = copy(call!(s, ω; params...))

Base.copy(s::ScatteringSolution) = ScatteringSolution(s.gω, copy_or_nothing.(s.leadsols), copy(s.matrix))
Base.copy(s::ScatteringMatrix) = ScatteringMatrix(copy.(s.PhiS), copy.(s.S))

copy_or_nothing(s::LeadSolution) =
    LeadSolution(copy(s.gh), copy(s.phi_a), copy(s.lambda_a), copy(s.ggpa), copy(s.prpg), copy(s.source))

copy_or_nothing(::Nothing) = nothing

## SelfEnergySchurSolver lead solution ##

# source = H_{CL}(Φₐ - gʳh₊ΦₐΛₐ⁻¹) and Γ = i(h₋gʳh₊ - (h₋gʳh₊)')
function solve_lead(sw::LeadWorkspace, solver::Union{SelfEnergySchurSolver,SelfEnergyCouplingSchurSolver})
    leadsol, ll = sw.leadsol, sw.ll
    _, HCL = coupling_to_from_lead(solver)
    h₋, h₊ = couplings_intralead(solver)
    gʳ = outgoing_gr(solver)
    λₐ, Φₐ, Φᵣ = incoming_λ(solver), incoming_Φ(solver), outgoing_Φ(solver)

    # Copying λₐ and Φₐ to lead solution
    copy!(leadsol.lambda_a, λₐ)
    copy!(leadsol.phi_a, Φₐ)

    # Building gʳh₊
    gʳh₊ = mul!(leadsol.gh, gʳ, h₊)   # gʳh₊

    # Building Γ = i(h₋gʳh₊ - (h₋gʳh₊)') Φᵣ'Γ and gʳΓΦₐ
    gʳΓΦₐ, Φᵣ´Γ = leadsol.ggpa, leadsol.prpg
    mul!(ll, h₋, gʳh₊, im, 0)         # ih₋gʳh₊
    Γ = (gʳΓΦₐ .= ll' .+ ll)          # gʳΓΦₐ temporarily holds Γ = ih₋gʳh₊ - i(h₋gʳh₊)'
    mul!(Φᵣ´Γ, Φᵣ', Γ)                # Φᵣ´Γ
    gʳΓ = mul!(ll, gʳ, Γ)             # gʳΓ = gʳ(i(h₋gʳh₊ - (h₋gʳh₊)')), aliases ll
    mul!(gʳΓΦₐ, gʳΓ, Φₐ)              # ll is now free

    # Building source
    mul!(ll, gʳh₊, Φₐ, -1, 0)
    ll ./= transpose(λₐ)            # -gʳh₊ΦₐΛₐ⁻¹
    ll .+= Φₐ                       # Φₐ - gʳh₊ΦₐΛₐ⁻¹
    mul!(leadsol.source, HCL, ll)   # H_{CL}(Φₐ - gʳh₊ΦₐΛₐ⁻¹)

    return leadsol
end

# fallback for non-Schur leads
solve_lead(_...) = nothing

function compute_scattering_matrix!(s::Scattering, Gω, leadsols, solvers)
    for j in eachindex(s.leadtmps)
        isnothing(s.leadtmps[j]) && continue
        compute_scattering_row!(s, j, Gω, leadsols, solvers)
    end
    return s.scattmp.smat
end

# φʳR = (iG₁₁Γ-1)Φₐ, φʳT = iG₁'₁ΓΦₐ
# r = √VᵣᵖPᵣRᵢPₐ⁺/√(-Vₐᵖ), tᵢⱼ = √VᵣᵢᵖPᵣᵢTᵢⱼPₐⱼ⁺/√(-Vₐⱼᵖ)
# with Γ = i(h₋gʳh₊ - (h₋gʳh₊)'), G₁₁ = gʳ + gʳH_{LC}GH_{CL}gʳ, G₁'₁ = g'ʳH_{L'C}GH_{CL}gʳ
# Then iG₁₁Γ = i(1 + gʳH_{LC}GH_{CL})gʳΓ so (iG₁₁Γ-1)Φₐ = i(1 + gʳH_{LC}GH_{CL})gʳΓΦₐ - Φₐ
# and iG₁₁Γ = ig´ʳH_{L´C}GH_{CL})gʳΓ, so iG₁´₁ΓΦₐ = ig´ʳH_{L´C}GH_{CL}gʳΓΦₐ
# Here G is the full central Greeen function
function compute_scattering_row!(s::Scattering, j, Gω, leadsols, solvers)
    # Diagonal block (reflection)
    solver, leadsol, leadtmp = solvers[j], leadsols[j], s.leadtmps[j]
    cl, lc, ll, smat = s.scattmp.cl_ij[j,j], s.scattmp.lc_i[j], leadtmp.ll, s.scattmp.smat
    PhiR, r, G00, Φₐ, gʳΓΦₐ, Φᵣ´Γ = smat.PhiS[j,j], smat.S[j,j], Gω[j,j], leadsol.phi_a, leadsol.ggpa, leadsol.prpg
    HLC, HCL = coupling_to_from_lead(solver)
    gʳ = outgoing_gr(solver)

    # Building G₁₁ abd φʳR
    mul!(cl, G00, HCL)
    mul!(lc, gʳ, HLC)
    copyto!(ll, I)
    mul!(ll, lc, cl, im, im)        # i(1 + gʳH_{LC}G₀₀H_{CL})
    copy!(PhiR, Φₐ)
    mul!(PhiR, ll, gʳΓΦₐ, 1, -1)    # φʳR = i(1 + gʳH_{LC}G₀₀H_{CL})gʳΓΦₐ - Φₐ = (iG₁₁Γ-1)Φₐ

    # reflection matrix r
    vaf = incoming_vfactors(solver)
    vrf = outgoing_vfactors(solver)
    mul!(r, Φᵣ´Γ, PhiR)
    @. r = vrf * r * vaf'

    # Off-diagonal blocks (transmissions)
    for i in 1:length(leadsols)
        (i == j || isnothing(leadsols[i])) && continue
        solver´, leadsol´ = solvers[i], leadsols[i]
        c´l, l´c´ = s.scattmp.cl_ij[i,j], s.scattmp.lc_i[i]
        PhiT, t, G0´0, Φᵣ´Γ = smat.PhiS[i,j], smat.S[i,j], Gω[i,j], leadsol´.prpg
        HL´C, _ = coupling_to_from_lead(solver´)
        g´ʳ = outgoing_gr(solver´)

        # Building G₁´₁ abd φʳR
        mul!(c´l, G0´0, HCL)
        mul!(l´c´, g´ʳ, HL´C)
        mul!(t, l´c´, c´l, im, 0)       # t temporarily holds ig´ʳH_{L´C}GH_{CL}
        mul!(PhiT, t, gʳΓΦₐ)            # φʳR = ig´ʳH_{L´C}GH_{CL}gʳΓΦₐ = iG₁´₁ΓΦₐ

        # transmission matrix t
        vrf´ = outgoing_vfactors(solver´)
        mul!(t, Φᵣ´Γ, PhiT)
        @. t = vrf´ * t * vaf'
    end

    return s
end


#endregion
