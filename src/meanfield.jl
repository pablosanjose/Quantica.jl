############################################################################################
# meanfield
#   designed to construct hartreefield and fockfield, such that
#     hartreefield[i] = ν * Q * Σ_k v_H(r_i-r_k) * tr(ρ[k,k]*Q)
#     fockfield[i,j]  = -v_F(r_i-r_j) * Q * ρ[i,j] * Q
#   where ν = ifelse(nambu, 1/2, 1), and Q is the charge matrix or [q 0; 0 -q] if nambu.
#   we precompute v_H^{ik} = \sum_n v_H(r_{i0} - r_{kn}), exploiting ρ translation symmetry
#region

struct MeanField{B,T,C<:CompressedOrbitalMatrix,S<:SparseMatrixCSC,F<:DensityMatrix}
    output::C
    potHartree::S
    potFock::S
    rho::F
    charge::B
    nambu::Bool
    is_nambu_rotated::Bool
    rowcol_ranges::NTuple{2,Vector{UnitRange{Int}}}
    onsite_tmp::Vector{T}
end

struct ZeroField end

#region ## Constructors ##

function meanfield(g::GreenFunction{T}, args...;
    potential = Returns(1), hartree = potential, fock = hartree,
    onsite = missing, charge = I, nambu::Bool = false, namburotation = missing,
    selector::NamedTuple = (; range = 0), selector_fock = selector, selector_hartree = selector, kw...) where {T}

    isempty(boundaries(g)) || argerror("meanfield does not currently support systems with boundaries")

    B = blocktype(hamiltonian(g))
    is_nambu_rotated´ = sanitize_nambu_rotated(namburotation, nambu, B)
    Q = sanitize_charge(charge, B, nambu, is_nambu_rotated´)

    hartreemodel = meanfield_models(g, onsite, hartree, selector_hartree)
    fockmodel = meanfield_models(g, onsite, fock, selector_fock)

    lat = lattice(hamiltonian(g))
    # scalar Hamiltonians that encode interactions between sites
    hFock = lat |> fockmodel
    hHartree = lat |> hartreemodel

    # SparseIndices(g, fockmodel) is like sitepairs(; ...) but using the selectors of all the model terms
    gs = g[SparseIndices(g, fockmodel)]
    gsQ = g[SparseIndices(g, fockmodel; kernel = Q)]
    rho = densitymatrix(gs, args...; kw...)

    # this drops zeros
    potHartree = T.(sum(unflat, harmonics(hHartree)))

    oaxes = orbaxes(call!_output(gs))
    rowcol_ranges = collect.(orbranges.(oaxes))
    onsite_tmp = Vector{T}(undef, length(last(rowcol_ranges)))

    # build potFock with identical axes as the output of rho
    cells_fock = cells(first(oaxes))
    hFock_slice = hFock[(; cells = cells_fock), (; cells = 0)]

    # this is important for the fast orbrange-based implementation of MeanField evaluation
    check_cell_order(hFock_slice, rho)
    # this does not drop zeros. Important to keep diagonal zeros
    potFock = convert(SparseMatrixCSC{T,Int}, parent(hFock_slice))

    encoder, decoder = nambu ? NambuEncoderDecoder(is_nambu_rotated´) : (identity, identity)
    S = typeof(encoder(zero(Q)))
    sparse_enc = similar(call!_output(gsQ), S)
    output = CompressedOrbitalMatrix(sparse_enc; encoder, decoder, hermitian = true)

    return MeanField(output, potHartree, potFock, rho, Q, nambu, is_nambu_rotated´, rowcol_ranges, onsite_tmp)
end

# Potential-based usage
function meanfield_models(::GreenFunction{T,E}, onsite, potential, selector) where {T,E}
    V = sanitize_potential(potential)
    U = onsite === missing ? T(V(zero(SVector{E,T}))) : T(onsite)
    U = potential === nothing ? zero(U) : U

    isfinite(U) || argerror("Onsite potential must be finite, consider setting `onsite`")

    # The sparse structure of hFock will be inherited by the evaluated mean field. Need onsite.
    model = hopping((r, dr) -> iszero(dr) ? U*I : T(V(dr))*I; selector..., includeonsite = true)
    return model
end

# Model-based usage
meanfield_models(::GreenFunction, _, model::TightbindingModel, _) = model + onsite(0I)
meanfield_models(::GreenFunction, _, model::AbstractModel, _) =
    argerror("Meanfield potentials cannot currently be parametric models. Use non-parametric `onsite`/`hopping` models instead.")

sanitize_potential(x::Real) = Returns(x)
sanitize_potential(x::Function) = x
sanitize_potential(x::Nothing) = Returns(0)
sanitize_potential(_) = argerror("Invalid potential: use a real number or a function of position")

sanitize_nambu_rotated(is_nambu_rotated, nambu, ::Type{<:SMatrix{2,2}}) = false
sanitize_nambu_rotated(is_nambu_rotated, nambu, ::Type{<:SMatrix{4,4}}) =
    nambu ? sanitize_nambu_rotated(is_nambu_rotated) : false
sanitize_nambu_rotated(is_nambu_rotated, nambu, B) =
    nambu ? nambu_dim_error(B) : false
sanitize_nambu_rotated(::Missing) =
    argerror("Must specify `namburotation` (true or false)")
sanitize_nambu_rotated(is_nambu_rotated::Bool) = is_nambu_rotated

function sanitize_charge(charge, B, nambu, is_rotated)
    Q = sanitize_charge(charge, B)
    ishermitian(Q) || argerror("Charge $Q should be Hermitian")
    nambu && check_nambu(Q, is_rotated)
    return Q
end

sanitize_charge(charge, B) = sanitize_block(B, charge)
sanitize_charge(charge, ::Type{<:SMatrixView}) = meanfield_multiorbital_error()

check_nambu(Q::S, is_rotated) where {S<:Union{SMatrix{2,2},SMatrix{4,4}}} =
    nambu_redundants(Q) ≈ nambu_adjoint_significants(Q, is_rotated) ||
    nambu_sym_error(Q)
check_nambu(Q::S, is_rotated) where {S<:SMatrix} = nambu_dim_error(S)
check_nambu(Q, is_rotated) = nambu_sym_error(Q)

nambu_dim_error(::Type{S}) where {N,S<:SMatrix{N,N}} =
    argerror("Nambu `meanfield` currently only understand 2×2 and 4×4 Nambu spaces, got $N×$N")
meanfield_multiorbital_error() =
    argerror("`meanfield` does not currently support systems with heterogeneous orbitals")
nambu_sym_error(Q) =
    argerror("Matrix $Q does not satisfy Nambu symmetry")

nambu_significants(mat::SMatrix{4,4}) = mat[:, SA[1,2]]
nambu_significants(mat::SMatrix{2,2}) = mat[:, 1]
nambu_redundants(mat::SMatrix{4,4}) = mat[:, SA[3,4]]
nambu_redundants(mat::SMatrix{2,2}) = mat[:, 2]

nambu_adjoint_significants(mat::SMatrix{N,N}, is_rotated) where {N} =
    nambu_adjoint_significants(nambu_significants(mat), is_rotated)

function nambu_adjoint_significants(lmat::SVector{2}, _)
    return SA[0 1; -1 0] * conj(lmat)
end

function nambu_adjoint_significants(lmat::SMatrix{4,2}, is_rotated)
    if is_rotated
        return -SA[0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0] * lmat * SA[0 -1; 1 0]
    else
        return -SA[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0] * lmat
    end
end

function NambuEncoderDecoder(is_nambu_rotated)
    encoder = nambu_significants
    decoder = smat -> [smat nambu_adjoint_significants(smat, is_nambu_rotated)]
    return encoder, decoder
end

function check_cell_order(hFock_slice, rho)
    opot = first(orbaxes(hFock_slice))
    orho = first(orbaxes(call!_output(rho.gs)))
    cells(opot) == cells(orho) || internalerror("meanfield: Cell order mismatch between potential and density matrix")
    return nothing
end

#endregion

#region ## API ##

charge(m::MeanField) = m.charge

hartree_matrix(m::MeanField) = m.potHartree

fock_matrix(m::MeanField) = parent(m.potFock)

isnambu(m::MeanField) = m.nambu

is_nambu_rotated(m::MeanField) = m.is_nambu_rotated

(m::MeanField)(args...; kw...) = copy(call!(m, args...; kw...))

function call!(m::MeanField{B}, args...; chopsmall = true, params...) where {B}
    Q, hartree_pot, fock_pot = m.charge, m.onsite_tmp, m.potFock
    rowrngs, colrngs = m.rowcol_ranges
    check_zero_mu(m, args...)
    ρ = m.rho(args...; params...)
    trρQ = maybe_nambufy_traces!(diag_real_tr_rho_Q(ρ, Q), m)
    mul!(hartree_pot, m.potHartree, trρQ)
    meanfield = m.output
    meanfield_parent = parent(meanfield)
    fill!(nonzeros(meanfield_parent), zero(eltype(meanfield_parent)))
    if chopsmall
        hartree_pot .= Quantica.chopsmall.(hartree_pot)  # this is a Vector
        nzs = nonzeros(parent(ρ))
        nzs .= Quantica.chopsmall.(nzs)
    end
    rows, cols, nzs = rowvals(fock_pot), axes(fock_pot, 2), nonzeros(fock_pot)
    for col in cols
        # 1/2 to compensate for the factor 2 from nambu trace
        viiQ = hartree_pot[col] * Q
        for ptr in nzrange(fock_pot, col)
            row = rows[ptr]
            row < col && ishermitian(meanfield) && continue   # skip upper triangle
            vij = nzs[ptr]
            ρij = view(ρ, rowrngs[row], colrngs[col])
            vQρijQ = vij * Q * sanitize_block(B, ρij) * Q
            if row == col
                meanfield_parent[row, col] = encoder(meanfield)(viiQ - vQρijQ)
            else
                meanfield_parent[row, col] = encoder(meanfield)(-vQρijQ)
            end
        end
    end
    return meanfield
end

check_zero_mu(_) = nothing
check_zero_mu(m, µ, _...) = !m.nambu || iszero(µ) ||
    argerror("Tried to evaluate a Nambu mean field at a nonzero µ = $µ")

# turns tr(ρ*Q) into 0.5*tr((ρ-SA[0 0; 0 I])Q) if nambu
function maybe_nambufy_traces!(traces, m::MeanField{B}) where {B}
    if isnambu(m)
        shift = tr(m.charge * hole_id(B))
        traces .-= shift
        traces .*= 0.5
    end
    return traces
end

diag_real_tr_rho_Q(ρ, Q) =
    [real(unsafe_trace_prod(Q, view(parent(ρ), rng, rng))) for rng in siteindexdict(orbaxes(ρ, 2))]

hole_id(::Type{<:SMatrix{2,2}}) = SA[0 0; 0 1]
hole_id(::Type{<:SMatrix{4,4}}) = SA[0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1]
hole_id(S) = nambu_dim_error(S)

## ZeroField

const zerofield = ZeroField()

(m::ZeroField)(args...; kw...) = m

Base.getindex(::ZeroField, _...) = 0.0 * I

#endregion

#endregion
