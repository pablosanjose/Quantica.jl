function pinverse(m::SMatrix{L,L´}) where {L,L´}
    qrm = qr(m, NoPivot())  # unlike qr(m), this produces a Matrix, has less latency
    pinvm = SMatrix{L´,L}(inv(qrm.R) * qrm.Q')
    return pinvm
end

# Make matrix square by appending (or prepending) independent columns if possible
function makefull(m::SMatrix{L,L´}) where {L,L´}
    Q = qr(m, NoPivot()).Q * I
    for i in 1:L * L´
        @inbounds Q[i] = m[i]         # overwrite first L´ cols with originals
    end
    return SMatrix{L,L}(Q)
end

# round to integers to preserve eltype
makefull(m::SMatrix{<:Any,<:Any,Int}) = round.(Int, makefull(float(m)))

# # Make matrix square by appending (or prepending) independent columns if possible
# function makefull(m::SMatrix{L,L´}, prepend = false) where {L,L´}
#     Q = qr(m, NoPivot()).Q * I
#     nnew = L * (L-L´)
#     nold = L * L´
#     if prepend
#         for i in 1:nnew
#             @inbounds Q[i] = Q[i + nold]  # new cols first (safe because it's sequential)
#         end
#         for i in 1:nold
#             @inbounds Q[nnew + i] = m[i]  # then originals
#         end
#     else
#         for i in 1:nold
#             @inbounds Q[i] = m[i]         # overwrite first L´ cols with originals
#         end
#     end
#     return SMatrix{L,L}(Q)
# end

# # round to integers to preserve eltype
# makefull(m::SMatrix{<:Any,<:Any,Int}, args...) = round.(Int, makefull(float(m), args...))