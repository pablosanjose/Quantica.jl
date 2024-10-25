############################################################################################
# meanfield
#   designed to construct hartreefield and fockfield, such that
#     hartreefield[i] = ν * Q * Σ_k v_H(r_i-r_k) * tr(ρ[k,k]*Q)
#     fockfield[i,j]  = -v_F(r_i-r_j) * Q * ρ[i,j] * Q
#   where ν = ifelse(nambu, 1/2, 1), and Q is the charge matrix or [q 0; 0 -q] if nambu.
#   we precompute v_H^{ik} = \sum_n v_H(r_{i0} - r_{kn}), exploiting ρ translation symmetry
#region

struct MeanField{B,T,S<:SparseMatrixCSC,H<:DensityMatrix,F<:DensityMatrix}
    potHartree::S
    potFock::S
    rhoHartree::H
    rhoFock::F
    charge::B
    rowcol_ranges::NTuple{2,Vector{UnitRange{Int}}}
    onsite_tmp::Vector{Complex{T}}
end

#region ## Constructors ##

function meanfield(g::GreenFunction{T,E}, args...;
    potential = Returns(1), hartree = potential, fock = hartree,
    onsite = missing, charge = I, nambu::Bool = false,
    selector::NamedTuple = (; range = 0), kw...) where {T,E}

    Vh = sanitize_potential(hartree)
    Vf = sanitize_potential(fock)
    Q = sanitize_charge(charge, blocktype(hamiltonian(g)))
    U = onsite === missing ? T(Vh(zero(SVector{E,T}))) : T(onsite)
    Uf = fock === nothing ? zero(U) : U

    isempty(boundaries(g)) || argerror("meanfield does not currently support systems with boundaries")
    isfinite(U) || argerror("Onsite potential must be finite, consider setting `onsite`")
    nambu && (!is_square(charge) || iseven(size(charge, 1))) && argerror("Invalid charge matrix for Nambu space")

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
    nambu && (nonzeros(potHartree) .*= T(1/2)) # to compensate for the factor 2 from nambu trace

    oaxes = orbaxes(call!_output(gsFock))
    rowcol_ranges = collect.(orbranges.(oaxes))
    onsite_tmp = similar(diag(parent(call!_output(gsHartree))))

    # build potFock with identical axes as the output of rhoFock
    cells_fock = cells(first(oaxes))
    hFock_slice = hFock[(; cells = cells_fock), (; cells = 0)]

    # this is important for the fast orbrange-based implementation of MeanField evaluation
    check_cell_order(hFock_slice, rhoFock)
    potFock = parent(hFock_slice)

    return MeanField(potHartree, potFock, rhoHartree, rhoFock, Q, rowcol_ranges, onsite_tmp)
end

# currying version
meanfield(; kw...) = g -> meanfield(g; kw...)
meanfield(x; kw...) = g -> meanfield(g, x; kw...)

sanitize_potential(x::Number) = Returns(x)
sanitize_potential(x::Function) = x
sanitize_potential(x::Nothing) = Returns(0)
sanitize_potential(_) = argerror("Invalid potential: use a number or a function of position")

sanitize_charge(B, t) = sanitize_block(t, B)
sanitize_charge(B, ::Type{<:SMatrixView}) = argerror("meanfield does not currently support systems with heterogeneous orbitals")

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

function (m::MeanField{B})(args...; params...) where {B}
    Q, hartree_pot, fock_pot = m.charge, m.onsite_tmp, m.potFock
    rowrngs, colrngs = m.rowcol_ranges
    trρQ = m.rhoHartree(args...; params...)
    mul!(hartree_pot, m.potHartree, diag(parent(trρQ)))
    meanfield = m.rhoFock(args...; params...)
    mf_parent = parent(meanfield)   # should be a sparse matrix
    rows, cols, nzs = rowvals(fock_pot), axes(fock_pot, 2), nonzeros(fock_pot)
    for col in cols
        viiQ = hartree_pot[col] * Q
        for ptr in nzrange(fock_pot, col)
            row = rows[ptr]
            vij = nzs[ptr]
            ρij = view(mf_parent, rowrngs[row], colrngs[col])
            vQρijQ = vij * Q * sanitize_block(B, ρij) * Q
            if row == col
                ρij .= viiQ - vQρijQ
            else
                ρij .= -vQρijQ
            end
        end
    end
    return meanfield
end

#endregion

#endregion
