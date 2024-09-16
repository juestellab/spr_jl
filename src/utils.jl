using HDF5
using StaticArrays


function save_spr(file_name, grid_type, coordiantes, spr_vals, central_sample_idx, t_start, index_offset, max_error, compute_times, overwrite=false)
    h5open(file_name, (overwrite ? "w" : "cw")) do f
        @assert grid_type in (:cartesian, :polar)
        grid = create_group(f, "grid")
        grid["grid_type"] = String(grid_type)
        if grid_type == :cartesian
            grid["grid_ordering"] = "x-y-z (imaging plane x-z)"
            grid["x"] = coordiantes[:x]
            grid["y"] = coordiantes[:y]
            grid["z"] = coordiantes[:z]
        elseif grid_type == :polar
            grid["grid_ordering"] = "phi-elevation-r (imaging plane phi-r)"
            grid["phi"] = coordiantes[:ϕ]
            grid["elevation"] = coordiantes[:elevation]
            grid["r"] = coordiantes[:r]
        end

        results = create_group(f, "results")
        results["spr_vals"] = spr_vals
        results["central_sample_idx"] = central_sample_idx
        results["t_start"] = t_start
        results["index_offset"] = index_offset
        results["max_error"] = max_error
        results["compute_times"] = compute_times
    end
end

function dict_from_python(py_opts)
    opts = Dict()
    for (key, val) in py_opts
        opts[Symbol(key)] = val
    end
    return opts
end

function get_cubature_opts(opts, pulse_peak, transducer_model)
    max_spr_val = pulse_peak * total_surface(transducer_model) / (4π * transducer_model.radius)
    cubature_opts = Dict()
    for key in (:atol, :rtol, :maxevals, :initdiv)
        cubature_opts[key] = opts[key]
    end
    if cubature_opts[:atol] == :auto
        @assert isfinite(max_spr_val) "max_spr_val is not finite, this can be caused by using an unfocused transducer"
        cubature_opts[:atol] = max_spr_val * cubature_opts[:rtol] * opts[:rtol_global]
    end
    return cubature_opts
end

grid_from_parameters(::Val{:cartesian}, x, y, z) = [SA[_x, _y, _z] for _y in y for _x in x for _z in z]
grid_from_parameters(::Val{:polar}, r, ϕ, elevation) = [SA[_r * sin(_ϕ), _e, _r * cos(_ϕ)] for _ϕ in ϕ for _e in elevation for _r in r]
grid_from_parameters(::Val{:adaptive_polar}, r, ϕ, elevation) = [SA[_r * sin(_ϕ), _e, _r * cos(_ϕ)] for _ϕ in ϕ for _e in elevation for _r in r]

index_array(::Val{:cartesian}, x, y, z) = [i*length(x)*length(z) + j*length(z) + l for l = eachindex(y), j = eachindex(x), i = eachindex(z)]
