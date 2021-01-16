# SingleShot1DGreensSolver

Goal: solve the semi-infinite Green's function [g₀⁻¹ -V'GV]G = 1, where GV = ∑ₖφᵢᵏ λᵏ (φ⁻¹)ᵏⱼ.
It involves solving retarded solutions to
    [(ωI-H₀)λ - V - V'λ²] φ = 0
We do a SVD of
    V = WSU'
    V' = US'W'
where U and W are unitary, and S is diagonal with perhaps some zero
singlular values. We write [φₛ, φᵦ] = U'φ, where the β sector has zero singular values,
    SPᵦ = 0
    U' = [Pₛ; Pᵦ]
    [φₛ; φᵦ] = U'φ = [Pₛφ; Pᵦφ]
    [Cₛₛ Cₛᵦ; Cᵦₛ Cᵦᵦ] = U'CU for any operator C
Then
    [λU'(ωI-H₀)U - U'VU - λ²U'VU] Uφ = 0
    U'VU = U'WS = [Vₛₛ 0; Vᵦₛ 0]
    U'V'U= S'W'U= [Vₛₛ' Vᵦₛ; 0 0]
    U'(ωI-H₀)U = U'(g₀⁻¹)U = [g₀⁻¹ₛₛ g₀⁻¹ₛᵦ; g₀⁻¹ᵦₛ g₀⁻¹ᵦᵦ] = ωI - [H₀ₛₛ H₀ₛᵦ; H₀ᵦₛ H₀ᵦᵦ]

The [λ(ωI-H₀) - V - λ²V'] φ = 0 problem can be solved by defining φχ = [φ,χ] = [φ, λφ] and
solving eigenpairs A*φχ = λ B*φχ, where
    A = [0 I; -V g₀⁻¹]
    B = [I 0; 0 V']
To eliminate zero or infinite λ, we reduce the problem to the s sector
    Aₛₛ [φₛ,χₛ] = λ Bₛₛ [φₛ,χₛ]
    Aₛₛ = [0 I; -Vₛₛ g₀⁻¹ₛₛ] + [0 0; -Σ₁ -Σ₀]
    Bₛₛ = [I 0; 0 Vₛₛ'] + [0 0; 0 Σ₁']
where
    Σ₀ = g₀⁻¹ₛᵦ g₀ᵦᵦ g₀⁻¹ᵦₛ + V'ₛᵦ g₀ᵦᵦ Vᵦₛ = H₀ₛᵦ g₀ᵦᵦ H₀ᵦₛ + Vₛᵦ g₀ᵦᵦ Vᵦₛ
    Σ₁ = - g₀⁻¹ₛᵦ g₀ᵦᵦ Vᵦₛ = H₀ₛᵦ g₀ᵦᵦ Vᵦₛ
    g₀⁻¹ₛₛ = ωI - H₀ₛₛ
    g₀⁻¹ₛᵦ = - H₀ₛᵦ
Here g₀ᵦᵦ is the inverse of the bulk part of g₀⁻¹, g₀⁻¹ᵦᵦ = ωI - H₀ᵦᵦ. To compute this inverse
efficiently, we store the Hessenberg factorization `hessbb = hessenberg(-H₀ᵦᵦ)` and use shifts.
Then, g₀ᵦᵦ H₀ᵦₛ = (hess + ω I) \ H₀ᵦₛ and g₀ᵦᵦ Vᵦₛ = (hess + ω I) \ Vᵦₛ.
Diagonalizing (Aₛₛ, Bₛₛ) we obtain the surface projection φₛ = Pₛφ of eigenvectors φ. The
bulk part is
    φᵦ = g₀ᵦᵦ (λ⁻¹Vᵦₛ - g₀⁻¹ᵦₛ) φₛ = g₀ᵦᵦ(λ⁻¹Vᵦₛ + H₀ᵦₛ) φₛ
so that the full φ's with non-zero λ read
    φ = U[φₛ; φᵦ] = [Pₛ' Pᵦ'][φₛ; φᵦ] = [Pₛ' + Pᵦ' g₀ᵦᵦ(λ⁻¹Vᵦₛ + H₀ᵦₛ)] φₛ = Z[λ] φₛ
    Z[λ] = [Pₛ' + Pᵦ' g₀ᵦᵦ(λ⁻¹Vᵦₛ + H₀ᵦₛ)]
We can complete the set with the λ=0 solutions, Vφᴿ₀ = 0 and the λ=∞ solutions V'φᴬ₀ = 0.
Its important to note that U'φᴿ₀ = [0; φ₀ᵦᴿ] and W'φᴬ₀ = [0; φ₀ᵦ´ᴬ]

By computing the velocities vᵢ = im * φᵢ'(V'λᵢ-Vλᵢ')φᵢ / φᵢ'φᵢ = 2*imag(χᵢ'Vφᵢ)/φᵢ'φᵢ we
classify φ into retarded (|λ|<1 or v > 0) and advanced (|λ|>1 or v < 0). Then
    φᴿ = Z[λᴿ]φₛᴿ = [φₛᴿ; φᵦᴿ]
    U'φᴿ = [φₛᴿ; φᵦᴿ]
    φᴬ = Z[λᴬ]φₛᴬ = [φₛᴬ; φᵦᴬ]
    Wφᴬ = [φₛ´ᴬ; φᵦ´ᴬ]  (different from [φₛᴬ; φᵦᴬ])
The square matrix of all retarded and advanced modes read [φᴿ φᴿ₀] and [φᴬ φᴬ₀].
We now return to the Green's functions. The right and left semiinfinite GrV and GlV' read
    GrV = [φᴿ φ₀ᴿ][λ 0; 0 0][φᴿ φ₀ᴿ]⁻¹ = φᴿ λ (φₛᴿ)⁻¹Pₛ
        (used [φᴿ φ₀ᴿ] = U[φₛᴿ 0; φᵦᴿ φᴿ₀ᵦ] => [φᴿ φ₀ᴿ]⁻¹ = [φₛᴿ⁻¹ 0; -φᵦᴿ(φₛᴿ)⁻¹ (φᴿ₀ᵦ)⁻¹]U')
    GlV´ = [φᴬ φ₀ᴬ][(λᴬ)⁻¹ 0; 0 0][φᴬ φ₀ᴬ]⁻¹ = φᴬ λ (φₛ´ᴬ)⁻¹Pₛ´
        (used [φᴬ φᴬ₀]⁻¹ = W[φₛ´ᴬ 0; φᵦ´ᴬ φ₀ᵦ´ᴬ] => [φᴬ φ₀ᴬ]⁻¹ = [(φₛ´ᴬ)⁻¹ 0; -φᵦ´ᴬ(φₛ´ᴬ)⁻¹ (φ₀ᵦ´ᴬ)⁻¹]W')

We can then write the local Green function G∞_{0} of the semiinfinite and infinite lead as
    Gr_{1,1} = Gr = [g₀⁻¹ - V'GrV]⁻¹
    Gl_{-1,-1} = Gl = [g₀⁻¹ - VGlV']⁻¹
    G∞_{0} = [g₀⁻¹ - VGlV' - V'GrV]⁻¹
    (GrV)ᴺ  = φᴿ (λᴿ)ᴺ  (φₛᴿ)⁻¹ Pₛ = χᴿ (λᴿ)ᴺ⁻¹  (φₛᴿ)⁻¹ Pₛ
    (GlV´)ᴺ = φᴬ (λᴬ)⁻ᴺ (φₛᴬ)⁻¹ Pₛₚ = φᴬ (λᴬ)¹⁻ᴺ(χₛᴬ)⁻¹ Pₛₚ
    φᴿ = [Pₛ' + Pᵦ' g₀ᵦᵦ ((λᴿ)⁻¹Vᵦₛ + H₀ᵦₛ)]φₛᴿ
    φᴬ = [Pₛ' + Pᵦ' g₀ᵦᵦ ((λᴬ)Vᵦₛ + H₀ᵦₛ)]φₛᴬ
where φₛᴿ and φₛᴬ are retarded/advanced eigenstates with eigenvalues λᴿ and λᴬ. Here Pᵦ is
the projector onto the uncoupled V sites, VPᵦ = 0, Pₛ = 1 - Pᵦ is the surface projector of V
and Pₛₚ is the surface projector of V'.

Defining
    GVᴺ = (GrV)ᴺ for N>0 and (GlV´)⁻ᴺ for N<0
we have
    Semiinifinite: G_{N,M} = (GVᴺ⁻ᴹ - GVᴺGV⁻ᴹ)G∞_{0}
    Infinite:      G∞_{N}  = GVᴺ G∞_{0}
Spelling it out
    Semiinfinite right  (N,M > 0): G_{N,M} = G∞_(N-M) - (GrV)ᴺ G∞_(-M) = [GVᴺ⁻ᴹ - (GrV)ᴺ (GlV´)ᴹ]G∞_0.
    At surface (N=M=1), G_{1,1} = Gr = (1- GrVGlV´)G∞_0 = [g₀⁻¹ - V'GrV]⁻¹
    Semiinfinite left   (N,M < 0): G_{N,M} = G∞_(N-M) - (GlV´)⁻ᴺ G∞_(-M) = [GVᴺ⁻ᴹ - (GlV´)⁻ᴺ(GrV)⁻ᴹ]G∞_0.
    At surface (N=M=-1), G_{1,1} = Gl = (1- GrVGlV´)G∞_0 = [g₀⁻¹ - VGlV']⁻¹
