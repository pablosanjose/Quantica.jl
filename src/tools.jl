function pinverse(m::SMatrix{L,L´}) where {L,L´}
    qrm = qr(m, NoPivot())  # unlike qr(m), this produces a Matrix, has less latency
    pinvm = SMatrix{L´,L}(inv(qrm.R) * qrm.Q')
    return pinvm
end

# Make matrix square by appending (or prepending) independent columns if possible
function makefull(m::SMatrix{L,L´}) where {L,L´}
    Q = qr(Matrix(m), NoPivot()).Q * I
    for i in 1:L * L´
        @inbounds Q[i] = m[i]         # overwrite first L´ cols with originals
    end
    return SMatrix{L,L}(Q)
end

# round to integers to preserve eltype
makefull(m::SMatrix{<:Any,<:Any,Int}) = round.(Int, makefull(float(m)))

rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)