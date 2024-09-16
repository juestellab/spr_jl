using SPR
using QuadGK

function analytic_SIR(p, a, R, c)
    h₀ = R * (1 - sqrt(1 - (a/R)^2))
    ρ = sqrt(p[1]^2 + p[2]^2)
    z = p[3]

    sinα = (z == R) ? 1 : (ρ/(R - z))/sqrt(1 + (ρ/(R-z))^2)
    h₁ = R / (2 * sqrt(ρ^2 + (R - z)^2))
    in_cone(ρ, z) = z != R && (ρ/abs(R - z) < a/sqrt(R^2 - a^2))

    t₁ = (R - sqrt(ρ^2 + (R-z)^2))/c
    t₂ = sqrt((a - ρ)^2 + (z - h₀)^2)/c
    t₃ = sqrt((a + ρ)^2 + (z - h₀)^2)/c
    t₄ = 2 * R / c - t₁

    ϕₘ(r) = begin
        θ = acos((z <= R ? 1 : -1) * (2*R*z - z^2 - ρ^2 - r^2)/(2 * r * sqrt( ρ^2 + (R - z)^2)))
        r₁ = r * sin(θ)
        ρ₁ = ρ + r*cos(θ)*sinα
        cos_res = sqrt(ρ^2 + (R - z)^2) / (r₁ * ρ^2)*(sqrt(ρ₁^2*(R - z)^2 - ρ^2*(a^2 - r₁^2 - ρ₁^2)) - ρ₁*(R-z))
        return (cos_res ≈ 1) ? zero(r) : acos(cos_res)
    end

    sir(t) = begin
        if t₂ < t < t₃
            return h₁ / π * ϕₘ(c * t)
        elseif in_cone(ρ, z) && ((t₁ < t < t₂ && z < R) || (t₃ < t < t₄ && z > R))
                return h₁
        else
            return zero(t)
        end
    end
    return sir
end


function analytic_SPR(z, a, R, c, σ)
    pulse = gaussian_pixel_model(σ)

    h₀ = R * (1 - sqrt(1 - (a/R)^2))

    t₁ = (R - abs(R - z))/c
    t₂ = sqrt(a^2 + (z - h₀)^2)/c
    t₃ = sqrt(a^2 + (z - h₀)^2)/c
    t₄ = 2 * R / c - t₁

    κ = σ^2 * R / abs(R - z)

    spr(t) = begin
        if z < R
            return κ/c * (pulse((t - t₁)*c) - pulse((t - t₂)*c))/2
        elseif z > R
            return κ/c * (pulse((t - t₃)*c) - pulse((t - t₄)*c))/2
        else
            return - h₀/c * (t*c - R) * pulse(t*c - R)/2
        end
    end
    return spr
end

function conv_sir(sir, pulse, t, c, pulse_length, rtol; order=10)
    return quadgk(τ -> sir(τ) * pulse((t - τ)*c),
                  t - 2*pulse_length/c, t + 2*pulse_length/c;
                  rtol=rtol, maxevals=1e6, order=order)[1]
end
