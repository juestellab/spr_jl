using ForwardDiff: derivative
using Roots
Nshape(pixel_model::Function) = scaled_time::Real -> -scaled_time * pixel_model(scaled_time)

gaussian_pixel_model(sigma::Real) = x -> exp(-0.5 * (x/sigma)^2) / (sqrt(2π) * sigma)
spherical_pixel_model(radius::Real) = x -> abs(x) < radius ? 1.0 : 0.0
soft_boundary_spherical_pixel_model(radius::Real,hardness::Real) = 
    x -> 1.0 / (1.0 + exp((abs(x) - radius) * hardness / radius))
gaussian_pixel_model_deriv(sigma, c) = t -> (t^2/sigma^2 - 1) * exp(-0.5 * (t/sigma)^2) / (sqrt(2π) * sigma)* c

function pulse_peak_position(pulse_function, sampling_step)
    x_max = find_zero(x -> derivative(y -> log(-pulse_function(abs(y))), x), sampling_step)
    return x_max
end

function pulse_length(pulse_function, peak_position, rel_threshold)
    pulse_peak = abs(pulse_function(peak_position))
    t_max = find_zero(x -> pulse_function(x) + pulse_peak * rel_threshold, 1.1 * peak_position)
    return t_max
end

function get_pulse_properties(pulse_function, sampling_step_size, rel_threshold)
    x_max = pulse_peak_position(pulse_function, sampling_step_size)
    t_max = pulse_length(pulse_function, x_max, rel_threshold)
    return t_max, abs(pulse_function(x_max))
end
