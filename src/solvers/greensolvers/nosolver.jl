
# dummy placeholder : will remove

struct NoSolver <:AbstractGreenSolver end
struct AppliedNoSolver <:AppliedGreenSolver end

apply(n::NoSolver, h::AbstractHamiltonian) = AppliedNoSolver()

