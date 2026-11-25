# extension stubs for EigenSolvers
get_eigen(s::AbstractEigenSolver, _...) =
    argerror("The eigensolver backend for EigenSolvers.$(nameof(typeof(s))) is not loaded. Did you first do `using $(ES.solverbackends(s))`?")

############################################################################################
# EigenSolvers module
#   An AbstractEigenSolver is defined by a set of kwargs for the eigensolver and a set of
#   methods AbstractMatrix -> Eigen associated to that AbstractEigenSolver
#region

module EigenSolvers

using FunctionWrappers: FunctionWrapper
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using FastLapackInterface: HermitianEigenWs, EigenWs
using LinearAlgebra: BlasComplex, checksquare
using Quantica: Eigen, I, lu, ldiv!, Hermitian
using Quantica: Quantica, AbstractEigenSolver, AbstractEigenWorkspace, MissingWorkspace,
                SVector, SMatrix, sanitize_eigen, call!_output, argerror, ishermitian_approx, isdiag
import Quantica: get_eigen

#endregion

############################################################################################
# AbstractEigensolvers and AbstractEigenWorkspaces
# A concrete s::AbstractEigenSolver must implement
#   - Quantica.get_eigen(s, mat) -> tuple (ε, Ψ) eigenpair
# A concrete ws::AbstractEigenWorkspace must implement
#   - input_matrix(ws) (must not alias call!_output(h) if eigen_ws! overwrites it)
#   - Quantica.eigen_ws!(ws, mat) -> Eigen, sorted, aliasing ws buffers
# A ws is not thread-safe, like call!(h, ...): use one per thread. It also doesn't need
# to implement minimal_callsafe_copy.
#region

# Extensions should add methods to get_eigen for their specific solver. It should return a
# tuple (ε, Ψ) where ε is the eigenvalues and Ψ the eigenvectors.
(s::AbstractEigenSolver)(mat) = sanitize_eigen(Quantica.get_eigen(s, mat)...)
(s::AbstractEigenSolver)(mat, ::MissingWorkspace) = s(mat)
(::AbstractEigenSolver)(mat, ws::AbstractEigenWorkspace) = Quantica.eigen_ws!(ws, mat)

# unless otherwise specified, the backend package matches the solver name
solverbackends(s::AbstractEigenSolver) = string(nameof(typeof(s)))

# fallback
is_thread_safe(::AbstractEigenSolver) = true

# The matrix the solver receives, by default an alias of h's call! output
# If we have a potential ws workspace, its internal input_matrix(ws) takes precedence
# Each concrete AbstractEigenWorkspace must define its input_matrix(ws)
input_matrix(::AbstractEigenSolver, h) = call!_output(h)
input_matrix(s::AbstractEigenSolver, h, ::MissingWorkspace) = input_matrix(s, h)
input_matrix(::AbstractEigenSolver, h, ws::AbstractEigenWorkspace) = input_matrix(ws)

# Each concrete AbstractEigenSolver can define its own workspace. Fallback: no workspace
eigen_workspace(::AbstractEigenSolver, h) = MissingWorkspace()

#### LinearAlgebra #####

struct LinearAlgebra{K} <: AbstractEigenSolver
    kwargs::K
end

LinearAlgebra(; kw...) = LinearAlgebra(NamedTuple(kw))

# no extension for LinearAlgebra, it is a strong dependency
# Hermitian eigen (complex ϵ, for type stability) if mat is Hermitian up to rounding and kwargs allow it
function Quantica.get_eigen(solver::LinearAlgebra, mat::AbstractMatrix{<:Number})
    kw = solver.kwargs
    # destructure: splatting an Eigen allocates
    if keys(kw) ⊆ (:sortby, :alg) && ishermitian_approx(mat)
        ϵ, ψ = Quantica.LinearAlgebra.eigen(Hermitian(mat); kw...)
        return complex.(ϵ), ψ
    else
        ϵ, ψ = Quantica.LinearAlgebra.eigen(mat; kw...)
        return ϵ, ψ
    end
end

## LinearAlgebra workspace add-on
# Without eigen kwargs (e.g. sortby, which LAPACK workspaces don't support), LinearAlgebra
# gets a LinearAlgebraWorkspace: dense input matrix, FastLapackInterface workspaces, plus
# output and permutation buffers, so that eigen_ws!(ws, mat) is allocation-free.
struct LinearAlgebraWorkspace{C<:Complex,H<:HermitianEigenWs,G<:EigenWs} <: AbstractEigenWorkspace
    mat::Matrix{C}
    hermitian_ws::H
    general_ws::G
    ϵs::Vector{C}       # complex eigenvalues for the Hermitian and isdiag branches
    ψs::Matrix{C}       # eigenvectors for the isdiag shortcut (LAPACK ones are in its ws)
    p::Vector{Int}      # permutation buffers for sorting
    p´::Vector{Int}
end

function LinearAlgebraWorkspace(mat::Matrix{C}) where {C<:BlasComplex}
    n = checksquare(mat)
    return LinearAlgebraWorkspace(mat, HermitianEigenWs(mat; vecs = true), EigenWs(mat),
        Vector{C}(undef, n), Matrix{C}(undef, n, n), Vector{Int}(undef, n), Vector{Int}(undef, n))
end

# LinearAlgebra.eigen doesn't like sparse Matrices as input, must convert
input_matrix(::LinearAlgebra, h) = Matrix(call!_output(h))
input_matrix(ws::LinearAlgebraWorkspace) = ws.mat

# only an ES.LinearAlgebra with no kwargs and BlasComplex eltype gets a LinearAlgebraWorkspace
# otherwise we fall back to MissingWorkspace
eigen_workspace(s::LinearAlgebra{@NamedTuple{}}, h) = linalg_workspace(input_matrix(s, h))

linalg_workspace(mat::Matrix{<:BlasComplex}) = LinearAlgebraWorkspace(mat)
linalg_workspace(_) = MissingWorkspace()  # no LAPACK workspaces for e.g. ComplexF16

# Hermitian eigen (real ϵs) if mat is Hermitian up to rounding, general eigen otherwise.
# Overwrites mat. Returns an Eigen sorted by (real, imag) that aliases the ws buffers, which
# are overwritten by the next call. The ishermitian_approx call is negligible vs eigen.
# Results are the LAPACK workspace buffers, except Hermitian eigenvalues, which are copied
# into ws.ϵs to make them complex, and the isdiag shortcut. Method of Quantica.eigen_ws!
function Quantica.eigen_ws!(ws::LinearAlgebraWorkspace, mat::AbstractMatrix)
    (; hermitian_ws, general_ws) = ws
    if isdiag(mat)          # as LinearAlgebra's no-LAPACK shortcut, exact unit eigenvectors
        herm = ishermitian_approx(mat)
        ϵs = ws.ϵs
        for i in eachindex(ϵs)
            x = mat[i, i]
            ϵs[i] = herm ? complex(real(x)) : x     # same type in both branches
        end
        ψs = copyto!(ws.ψs, I)
        sorteigs_ws!(ws, ϵs, ψs)
    elseif ishermitian_approx(mat)
        ϵs´, ψs = Quantica.eigen_ws!(hermitian_ws, mat)     # real ϵs´, sorted by LAPACK
        ϵs = (ws.ϵs .= ϵs´)                                 # complex, for type stability
    else
        ϵs, ψs = Quantica.eigen_ws!(general_ws, mat)        # complex, unsorted
        sorteigs_ws!(ws, ϵs, ψs)
    end
    return Eigen(ϵs, ψs)
end

# sort ϵs and the columns of ψs by (real, imag), as LinearAlgebra's sorteig!, using the ws
# permutation buffers
function sorteigs_ws!(ws::LinearAlgebraWorkspace, ϵs, ψs)
    (; p, p´) = ws
    by = λ -> (real(λ), imag(λ))
    if !issorted(ϵs; by)
        sortperm!(p, ϵs; alg = QuickSort, by)
        Quantica.permute!!(ϵs, copy!(p´, p))
        Quantica.permutecols!!(ψs, p)
    end
    return ϵs, ψs
end

#### Arpack #####

struct Arpack{K<:NamedTuple} <: AbstractEigenSolver
    kwargs::K
end

Arpack(; kw...) = Arpack(NamedTuple(kw))

# See https://github.com/JuliaLinearAlgebra/Arpack.jl/issues/86
is_thread_safe(::Arpack) = false

#### KrylovKit #####

struct KrylovKit{P<:Tuple,K<:NamedTuple} <: AbstractEigenSolver
    params::P
    kwargs::K
end

KrylovKit(params...; kw...) = KrylovKit(params, NamedTuple(kw))

#### ArnoldiMethod #####

struct ArnoldiMethod{K<:NamedTuple} <: AbstractEigenSolver
    kwargs::K
end

ArnoldiMethod(; kw...) = ArnoldiMethod(NamedTuple(kw))

#### ShiftInvert ####

struct ShiftInvert{T,E<:AbstractEigenSolver} <: AbstractEigenSolver
    eigensolver::E
    origin::T
end

solverbackends(s::ShiftInvert) = "LinearMaps"

#endregion

end # module

const ES = EigenSolvers

#endregion
