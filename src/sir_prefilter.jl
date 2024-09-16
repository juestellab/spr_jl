using FFTW
using BSplineKit

function sampled(fₓ, L)
    fₛ = zero(fₓ)
    fₛ[1:L:length(fₓ)] .= fₓ[1:L:length(fₓ)]
    return fₛ
end

function compute_prefilter(λ, L, Δx, n_steps)
    total_steps = (n_steps + 1 ) * L - (L - 1) ÷ 2
    x_half = range(0, length = total_steps, step=Δx/L)
    if iseven(L)
        x = vcat(-reverse(x_half[2:end]), x_half[1:(end-1)]) |> ifftshift
    else
        x = vcat(-reverse(x_half[2:end]), x_half) |> ifftshift
    end
    @assert length(x) % L == 0

    λₓ = λ.(x)
    λₛ = sampled(λₓ, L)
    Λₖ = fft(λₓ)

    Γₖ = Λₖ./fft(λₛ)
    aₓ = real.(ifft(abs2.(Γₖ)))
    aₛ = sampled(aₓ, L)

    ϕₓ = conj(Γₖ ./ fft(aₛ)) |> ifft .|> real
    Γₓ = ifft(Γₖ) .|> real
    return fftshift(x), fftshift(ϕₓ), fftshift(Γₓ)
end


function splines(loc, val)
    interpolator = interpolate(loc, val, BSplineOrder(3))
    return x -> ((x > minimum(loc)) && x < maximum(loc)) ? interpolator(x) : 0
end


function prefilter(t_res, L=15, sharpness_factor=2)
    λ = gaussian_pixel_model(t_res/sharpness_factor)
    x, p, f = compute_prefilter(λ, L, t_res, 10) 
    normalization = sum(p)*t_res/L
    return splines(x, p./normalization), splines(x, f)
end
