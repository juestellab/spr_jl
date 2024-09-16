using LinearAlgebra
using Random
using Distributed
using SharedArrays
using StaticArrays
using HCubature
using ProgressMeter


####################################################################################################
#Integration
####################################################################################################
function integrand(t::T, r::SVector{3,T}, c::T, pulse_shape::Function, transducer_model::AbstractTransducer{T}) where {T<:AbstractFloat}
    dist(q1::Real, q2::Real) = norm(r .- unsafe_position(transducer_model, q1, q2))
    function temporal_impulse(q)
        d = dist(q...)
        pulse_shape(t * c - d) / (4π * d * c) * dS(transducer_model, q...)
    end
    return temporal_impulse
end

function compute_spr_point(t, r, c, pulse_shape, transducer_model, cubature_opts)
        response = integrand(t, r, c, pulse_shape, transducer_model)
        return hcubature(response, transducer_model.min_corner, transducer_model.max_corner; cubature_opts...)
end

"""
    single_spr(r::SVector{3,T}, c::T, transducer_model::AbstractTransducer{T}, pulse_shape, time_range, cubature_opts=Dict()) where {T}
Compute the SPR for a single position in the FOV
"""
function single_spr(r::SVector{3,T}, c::T, transducer_model::AbstractTransducer{T}, pulse_shape, time_range, cubature_opts=Dict()) where {T}
    error = T(0)
    total_squared_error = T(0)
    result = Vector{eltype(r)}(undef, length(time_range))
    t_start = time()
    for (i, t) in enumerate(time_range)
        result[i], err = compute_spr_point(t, r, c, pulse_shape, transducer_model, cubature_opts)
        error = max(error, err)
        total_squared_error += err^2
    end
    return (result, error, total_squared_error/length(time_range), time() - t_start)
end

"""
Low-level interface to compute the SPR for a set of pixels in the field of view. 
If multiple workers are availible, the computation will be distributed to all.

Parameters
================================================================================
fov_points::Array       3D positions of each point
transducer_model::AbstractTransducer    Specifies geometry of the transducer
pulse_shape::Function   Emitted pulse, centered around t=0
f_sampling::Real        Sampling rate of the SPR
c::Real                 Speed of sound (in the coupling medium)
computation_parameters: Dictionary containing options for the cubature and extent of the SPR.
    (For defaults see spr_julia.default_computation_params)
    :threshold  Temporal expent of the SPR for each fov point is limited to first time it exceeds
                max(spr)*threshold and the last time it does.
    :rtol       Relative tolarance of the cubature.
    :rtol_global        If :atol is set to :auto, this is used to calculate a reasonable :atol
    :atol       Absolute tolerance of the cubature. Can be either a positive,
                real number or :auto. If :auto, the absolute tolerance will be set
                to the estimated maximum value of the SPR * rtol * rtol_global
    :maxevals   Maximal evaluation of the integrand, until the cubature is terminated.
    :initdiv    Number of divisions of the transducer area, that the cubature starts with.

Returns
================================================================================
collective_spr_vals::Array              Contigous array containing the SPR in the order of fov_points
offset_lims_to_central_sample::Array  
index_offset::Array                     Array of length(fov_points)+1. SPR of the i-th fov_point
                                        can be indexed by index_offset[i]:index_offset[i+1]
max_error::Array                        Maximal estimated error of the SPR over time samples for 
                                        each fov point
compute_times::Array                    Time in seconds the computation for each point took
"""
function compute_spr(fov_points, transducer_model::AbstractTransducer, pulse_shape::Function,
                     f_sampling, c, computation_parameters=Dict(), verbose=true)
    @info "Setup calcuation..."
    dtype = eltype(fov_points[1])
    c = convert(dtype, c)
    fov_points = (eltype(fov_points) <: SArray{Tuple{3}}) ? fov_points : [SVector{3}(r) for r in fov_points]
    computation_parameters = merge(default_computation_params, computation_parameters)

    pulse_width, pulse_peak = get_pulse_properties(pulse_shape, 1 / f_sampling, computation_parameters[:threshold])
    
    time_ranges, index_offset, central_sample_idx = get_time_information(fov_points, pulse_width,
                                                                         transducer_model, f_sampling, c)
    cubature_opts = get_cubature_opts(computation_parameters, pulse_peak, transducer_model)
    collective_spr_vals, max_error, mean_squared_error, compute_times = compute_spr_lowlevel(
        fov_points, time_ranges, index_offset, transducer_model, pulse_shape, c, cubature_opts, verbose)
    t_start = minimum.(time_ranges)
    return collective_spr_vals, central_sample_idx, t_start, index_offset, max_error, mean_squared_error, compute_times
end


function compute_spr_lowlevel(fov_points, time_ranges, index_offset, transducer_model::AbstractTransducer, pulse_shape::Function,
                            c, cubature_opts=Dict(), verbose=true)
    dtype = eltype(fov_points[1])
    # Initialize required arrays
    max_error = SharedArray{dtype,1}(length(fov_points))
    mean_squared_error = SharedArray{dtype,1}(length(fov_points))
    compute_times = SharedArray{dtype,1}(length(fov_points))
    collective_spr_vals = SharedArray{dtype,1}(index_offset[end])
    spr_index(i) = (1 + index_offset[i]):index_offset[i+1]

    if verbose
        @info "Computing SPR for $(length(fov_points)) points"
        @info "Time samples: Total $(length(collective_spr_vals))\t\t
        Average per point $(length(collective_spr_vals)/length(fov_points))\t\t 
        Min/Max $(extrema(length.(time_ranges)))"
        @showprogress pmap(eachindex(fov_points)) do i
            collective_spr_vals[spr_index(i)], max_error[i], mean_squared_error[i], compute_times[i] =
                single_spr(fov_points[i], c, transducer_model, pulse_shape, time_ranges[i], cubature_opts)
        end
    else
        pmap(eachindex(fov_points)) do i
            collective_spr_vals[spr_index(i)], max_error[i], mean_squared_error[i], compute_times[i] =
                single_spr(fov_points[i], c, transducer_model, pulse_shape, time_ranges[i], cubature_opts)
        end
    end
    return collective_spr_vals, max_error, mean_squared_error, compute_times
end

####################################################################################################
#Calculate pulse limits
####################################################################################################
function get_time_information(fov_points, pulse_width, transducer_model, f_sampling, c)
    dtype = eltype(fov_points[1])
    _range_type = Base.TwicePrecision{dtype}
    time_ranges = SharedArray{StepRangeLen{dtype, _range_type, _range_type, Int64}}(length(fov_points))
    center_idx = SharedArray{dtype}(length(fov_points))

    @sync @distributed for i = eachindex(fov_points)
        time_ranges[i], center_idx[i] = get_time_range(fov_points[i], c, 1.0 / f_sampling, pulse_width, transducer_model, (20, 20))
    end
    index_offset = [0, cumsum(length.(time_ranges))...]
    return time_ranges, index_offset, center_idx
end


function get_time_range(r, c, Δt, recording_offset, transducer_model, sampling)
    q = surface_samples(transducer_model, sampling)

    r_min = Inf
    r_max = 0.0
    center_time = norm(r) / c
    for q1 in q[1], q2 in q[2]
        r_transducer = unsafe_position(transducer_model, q1, q2)
        dist = norm(r - r_transducer)
        r_min = min(r_min, dist)
        r_max = max(r_max, dist)
    end
    @assert r_min < r_max
    t_min = (r_min - recording_offset) / c
    t_max = (r_max + recording_offset) / c

    min_sample_idx = fld(t_min - center_time, Δt)
    start_time = center_time + min_sample_idx * Δt
    steps = Int(cld(t_max - t_min, Δt))
    return range(start=start_time, step=Δt, length=steps), -Int(min_sample_idx) + 1
end
