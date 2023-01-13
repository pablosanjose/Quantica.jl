
# dummy placeholder : will remove

struct AppliedNoSolver <:AppliedGreenSolver end

apply(n::GS.NoSolver, h::AbstractHamiltonian) = AppliedNoSolver()

