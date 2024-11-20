############################################################################################
# meanfield
#   designed to construct hartreefield and fockfield, such that
#     hartreefield[i] = ν * Q * Σ_k v_H(r_i-r_k) * tr(ρ[k,k]*Q)
#     fockfield[i,j]  = -v_F(r_i-r_j) * Q * ρ[i,j] * Q
#   where ν = ifelse(nambu, 1/2, 1), and Q is the charge matrix or [q 0; 0 -q] if nambu.
#   we precompute v_H^{ik} = \sum_n v_H(r_{i0} - r_{kn}), exploiting ρ translation symmetry
#region

struct MeanField{B,T,C<:CompressedOrbitalMatrix,S<:SparseMatrixCSC,H<:DensityMatrix,F<:DensityMatrix}
    output::C
    potHartree::S
    potFock::S
    rhoHartree::H
    rhoFock::F
    charge::B
    nambu::Bool
    is_nambu_rotated::Bool
    rowcol_ranges::NTuple{2,Vector{UnitRange{Int}}}
    onsite_tmp::Vector{Complex{T}}
end

struct ZeroField end

#region ## Constructors ##

function meanfield(g::GreenFunction{T,E}, args...;
    potential = Returns(1), hartree = potential, fock = hartree,
    onsite = missing, charge = I, nambu::Bool = false, namburotation = missing,
    selector::NamedTuple = (; range = 0), kw...) where {T,E}

    Vh = sanitize_potential(hartree)
    Vf = sanitize_potential(fock)
    is_nambu_rotated´ = sanitize_nambu_rotated(namburotation, nambu)
    Q = sanitize_charge(charge, blocktype(hamiltonian(g)), nambu, is_nambu_rotated´)
    U = onsite === missing ? T(Vh(zero(SVector{E,T}))) : T(onsite)
    Uf = fock === nothing ? zero(U) : U

    isempty(boundaries(g)) || argerror("meanfield does not currently support systems with boundaries")
    isfinite(U) || argerror("Onsite potential must be finite, consider setting `onsite`")

    gsHartree = g[diagonal(; cells = 0, kernel = Q)]
    rhoHartree = densitymatrix(gsHartree, args...; kw...)
    gsFock = g[sitepairs(; selector..., includeonsite = true)]
    rhoFock = densitymatrix(gsFock, args...; kw...)

    lat = lattice(hamiltonian(g))
    # The sparse structure of hFock will be inherited by the evaluated mean field. Need onsite.
    hFock = lat |> hopping((r, dr) -> iszero(dr) ? Uf : Vf(dr); selector..., includeonsite = true)
    hHartree = (Uf == U && Vh === Vf) ? hFock :
        lat |> hopping((r, dr) -> iszero(dr) ? U : Vh(dr); selector..., includeonsite = true)

    potHartree = sum(unflat, harmonics(hHartree))

    oaxes = orbaxes(call!_output(gsFock))
    rowcol_ranges = collect.(orbranges.(oaxes))
    onsite_tmp = similar(diag(parent(call!_output(gsHartree))))

    # build potFock with identical axes as the output of rhoFock
    cells_fock = cells(first(oaxes))
    hFock_slice = hFock[(; cells = cells_fock), (; cells = 0)]

    # this is important for the fast orbrange-based implementation of MeanField evaluation
    check_cell_order(hFock_slice, rhoFock)
    potFock = parent(hFock_slice)

    encoder, decoder = nambu ? NambuEncoderDecoder(is_nambu_rotated´) : (identity, identity)
    S = typeof(encoder(zero(Q)))
    output = call!_output(g[sitepairs(; selector..., includeonsite = true, kernel = Q)])
    sparse_enc = similar(output, S)
    output = CompressedOrbitalMatrix(sparse_enc; encoder, decoder, hermitian = true)

    return MeanField(output, potHartree, potFock, rhoHartree, rhoFock, Q, nambu, is_nambu_rotated´, rowcol_ranges, onsite_tmp)
end

sanitize_potential(x::Number) = Returns(x)
sanitize_potential(x::Function) = x
sanitize_potential(x::Nothing) = Returns(0)
sanitize_potential(_) = argerror("Invalid potential: use a number or a function of position")

sanitize_nambu_rotated(is_nambu_rotated, nambu) =
    nambu ? sanitize_nambu_rotated(is_nambu_rotated) : false
sanitize_nambu_rotated(::Missing) =
    argerror("Must specify `namburotation` (true or false)")
sanitize_nambu_rotated(is_nambu_rotated::Bool) = is_nambu_rotated

function sanitize_charge(charge, B, nambu, is_rotated)
    Q = sanitize_charge(charge, B)
    nambu && check_nambu(Q, is_rotated)
    return Q
end

sanitize_charge(charge, B) = sanitize_block(B, charge)
sanitize_charge(charge, ::Type{<:SMatrixView}) =
    argerror("meanfield does not currently support systems with heterogeneous orbitals")

check_nambu(Q::S, is_rotated) where {S<:Union{SMatrix{2,2},SMatrix{4,4}}} =
    nambu_redundants(Q) ≈ nambu_adjoint_significants(Q, is_rotated) ||
    argerror("Matrix $Q does not satisfy Nambu symmetry")
check_nambu(::SMatrix{N,N}, is_rotated) where {N} =
    argerror("Quantica currently only understand 2×2 and 4×4 Nambu spaces, got $N×$N")
check_nambu(Q, is_rotated) = argerror("$Q does not satisfy Nambu symmetry")

nambu_significants(mat::SMatrix{4,4}) = mat[:, SA[1,2]]
nambu_significants(mat::SMatrix{2,2}) = mat[:, 1]
nambu_redundants(mat::SMatrix{4,4}) = mat[:, SA[3,4]]
nambu_redundants(mat::SMatrix{2,2}) = mat[:, 2]

nambu_adjoint_significants(mat::SMatrix{N,N}, is_rotated) where {N} =
    nambu_adjoint_significants(nambu_significants(mat), is_rotated)

function nambu_adjoint_significants(lmat::SVector{2}, _)
    return -SA[0 1; 1 0] * conj(lmat)
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

function check_cell_order(hFock_slice, rhoFock)
    opot = first(orbaxes(hFock_slice))
    orho = first(orbaxes(call!_output(rhoFock.gs)))
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
    trρQ = m.rhoHartree(args...; params...)
    mul!(hartree_pot, m.potHartree, diag(parent(trρQ)))
    ρFock = m.rhoFock(args...; params...)
    meanfield = m.output
    mf_parent = parent(meanfield)
    fill!(mf_parent, zero(eltype(mf_parent)))
    if chopsmall
        hartree_pot .= Quantica.chopsmall.(hartree_pot)  # this is a Vector
        nzs = nonzeros(parent(ρFock))
        nzs .= Quantica.chopsmall.(nzs)
    end
    rows, cols, nzs = rowvals(fock_pot), axes(fock_pot, 2), nonzeros(fock_pot)
    for col in cols
        # 1/2 to compensate for the factor 2 from nambu trace
        viiQ = ifelse(m.nambu, 0.5*hartree_pot[col], hartree_pot[col]) * Q
        for ptr in nzrange(fock_pot, col)
            row = rows[ptr]
            row > col && ishermitian(meanfield) && continue   # skip upper triangle
            vij = nzs[ptr]
            ρij = view(ρFock, rowrngs[row], colrngs[col])
            vQρijQ = vij * Q * sanitize_block(B, ρij) * Q
            if row == col
                mf_parent[row, col] = encoder(meanfield)(viiQ - vQρijQ)
            else
                mf_parent[row, col] = encoder(meanfield)(-vQρijQ)
            end
        end
    end
    return meanfield
end

## ZeroField

const zerofield = ZeroField()

(m::ZeroField)(args...; kw...) = m

Base.getindex(::ZeroField, _...) = 0.0 * I

#endregion

#endregion
