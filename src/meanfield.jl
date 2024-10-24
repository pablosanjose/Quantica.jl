# ############################################################################################
# # meanfield
# #   designed to construct hfield and ffield, such that
# #     hfield[i] = ν * Σ_k v_H(r_i-r_k) * tr(ρ[k,k]*Q)
# #     ffield[i,j] = v_F(r_i-r_j) * Q * ρ[i,j] * Q
# #   where ν = ifelse(nambu, 1/2, 1), and Q is the charge matrix or [q 0; 0 -q] if nambu.
# #   we precompute v_H^{ik} = \sum_n v_H(r_{i0} - r_{kn}), exploiting ρ translation symmetry
# #region

# struct MeanField{T,O<:OrbitalSliceMatrix{T},H,F,Q,B}
#     Vhmat::SparseMatrixCSC{T,Int}
#     Vfmat::O
#     hrho::H
#     frho::F
#     charge::Q
#     blocktype::B
# end

# struct EvaluatedMeanField{H<:OrbitalSliceVector,F<:OrbitalSliceMatrix}
#     hfield::H
#     ffield::F
# end

# #region ## Constructors ##

# function meanfield(g::GreenFunction{T}, args...;
#     potential = Returns(0), hartreepotential = potential, fockpotential = hartreepotential,
#     charge = I, nambu::Bool = false, selector = (; range = 0), kw...) where {T}

#     Vh = sanitize_potential(hartreepotential)
#     Vf = sanitize_potential(fockpotential)

#     B = blocktype(hamiltonian(g))
#     lat = lattice(hamiltonian(g))
#     hFock = lat |> hopping((r, dr) -> Vf(dr); selector..., includeonsite = true)
#     hHartree = Vh === Vf ? hFock :
#         lat |> hopping((r, dr) -> Vh(dr); selector..., includeonsite = true)

#     ldst, lsrc = ham_to_latslices(hFock)
#     odst, osrc = sites_to_orbs(ldst, g), sites_to_orbs(lsrc, g)
#     ρhartree = densitymatrix(g[diagonal(osrc, kernel = charge)], args...; kw...)
#     ρfock = densitymatrix(g[odst, osrc], args...; kw...)

#     Vhmat = sum(unflat, harmonics(hHartree))
#     nambu && (nonzeros(Vhmat) .*= T(1/2))
#     Vfmat = hFock[ldst, lsrc]

#     return MeanField(Vhmat, Vfmat, ρhartree, ρfock, charge, B)
# end

# sanitize_potential(x::Number) = Returns(x)
# sanitize_potential(x::Function) = x

# sanitize_charge(x, nambu) = nambu ? SA[x 0*x; 0*x -x] : x

# #endregion

# #region ## API ##


# function (m::MeanField{T})(args...; params...) where {T}
#     Q, B = m.charge, m.blocktype
#     trρQ = m.hrho(args...; params...)
#     QvtrρQ = (m.Vhmat * parent(trρQ)) .* Ref(Q)
#     hfield = OrbitalSliceVector(QvtrρQ, orbaxes(trρQ))
#     ffield = m.frho(args...; params...)
#     if !isempty(ffield)
#         oi, oj = orbaxes(ffield)
#         for sj in sites(oj, B), si in sites(oi, B)  # si, sj are CellSitePos
#             vij = T(only(view(m.Vfmat, CellSite(si), CellSite(sj)))) # a scalar
#             view(ffield, si, sj) .= vij * (Q * ffield[si, sj] * Q)
#         end
#     end
#     return EvaluatedMeanField(hfield, ffield)
# end

# Base.view(m::EvaluatedMeanField, i) = view(m.hfield, i)
# Base.view(m::EvaluatedMeanField, i, j) = view(m.ffield, i, j)

# Base.getindex(m::EvaluatedMeanField, i) = m.hfield[i]
# Base.getindex(m::EvaluatedMeanField, i, j) = m.ffield[i, j]
# #endregion

# #endregion
