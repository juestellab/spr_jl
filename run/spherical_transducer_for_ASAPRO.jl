using Distributed
nprocs() == 1 && addprocs(; exeflags="--project=@.")

using SPR
using StaticArrays
using HDF5
using Printf

@everywhere using SPR
include("spherical_transducer_analytical.jl")


function main(rtol_list, rtol_global_list, initdiv_list, f)
    path = "spr_spherical_transducer_out_of_axis.h5"
    c = 1500

    diameter = 20e-3
    r_focus = 20e-3
    transducer = SphericalTransducer(radius=r_focus, diameter=diameter)

    r_gauss = 50e-6
    pulse = Nshape(gaussian_pixel_model(r_gauss))
    fov_resolution = 0.2e-3
    fov = init_fov((5e-3, 35e-3), (fov_resolution, diameter/2), fov_resolution, r_focus)

    store_setup(path, r_focus, diameter, f, r_gauss, c, fov)

    central_sample_idx = missing
    idx_offset = missing
    computation_params = copy(SPR.default_computation_params)
    for rtol in rtol_list, rtol_global in rtol_global_list, initdiv in initdiv_list
        println("rtol = $rtol, rtol_global = $rtol_global, initdiv = $initdiv")
        computation_params[:rtol] = 10^-Float64(rtol)
        computation_params[:rtol_global] = 10^-Float64(rtol_global)
        computation_params[:initdiv] = initdiv

        spr_vals, _central_sample_idx, _, _idx_offset, max_err, total_squared_err, compute_times = compute_spr(fov, transducer, pulse,
                                                                                       f, c, computation_params)
        if ismissing(central_sample_idx) || ismissing(idx_offset)
            central_sample_idx = _central_sample_idx
            idx_offset = _idx_offset
        end
        store_results(path, rtol, rtol_global, initdiv, spr_vals, max_err, total_squared_err, compute_times)
    end

    spr_analytic, sir_analytic = compute_analytic(diameter, r_focus, c, fov, pulse,
                                                  central_sample_idx, idx_offset, f, computation_params[:threshold])

    store_analytic(path, spr_analytic, sir_analytic)

    h5open(path, "r+") do file
        setup = open_group(file, "setup")
        setup["central_sample_idx"] = central_sample_idx
        setup["idx_offset"] = idx_offset
    end
end


################################################################################
# SAVING SETUP AND RESULTS
################################################################################
function store_setup(path, r_focus, diameter, f, r_gauss, c, fov)
    h5open(path, "w") do file
        setup = create_group(file, "setup")
        setup["transducer_focus"] = r_focus
        setup["transducer_diameter"] = diameter
        setup["sampling_frequency"] = f
        setup["pixel_size"] = r_gauss
        setup["speed_of_sound"] = c
        setup["fov_points"] = fov
    end
end

function store_results(path, rtol, rtol_global, initdiv, spr_vals, max_err, total_squared_err, compute_times)
    h5open(path, "r+") do file
        results = create_group(file, "results_logrtol=-$(@sprintf("%.1e", rtol))_logrtol_global=-$(@sprintf("%.1e", rtol_global))_initdiv=$initdiv")
        results["spr_vals"] = spr_vals
        results["max_error"] = max_err
        results["total_squared_error"] = total_squared_err
        results["compute_times"] = compute_times
    end
end

function store_analytic(path, spr_analytic, sir_analytic)
    h5open(path, "r+") do file
        results = create_group(file, "results_analytic")
        results["spr_analytic"] = spr_analytic
        results["sir_analytic"] = sir_analytic
    end
end

################################################################################
# INITIALIZATION
################################################################################
function init_fov(z_bounds, y_bounds, fov_resolution, r_focus)
    z_range = z_bounds[1]:fov_resolution:z_bounds[2] |> collect
    idx_z_focus = findfirst(z -> z == r_focus, z_range)
    !isnothing(idx_z_focus) && deleteat!(z_range, idx_z_focus)
    y_range = y_bounds[1]:fov_resolution:y_bounds[2]
    return [SA[0., y, z] for z in z_range, y in y_range]
end

################################################################################
# COMPUTATION
################################################################################
function compute_analytic(diameter, r_focus, c, fov, pulse, central_sample_idx, idx_offset, f, pulse_threshold)
    spr_analytic = Float64[]
    sir_analytic = Float64[]
    pulse_length, _ = SPR.get_pulse_properties(pulse, 1/f, pulse_threshold)
    for (i, p) in enumerate(fov)
        t_samples = range(start=sqrt(sum(abs2, p))/c - (central_sample_idx[i] -1)/f,
                          length=idx_offset[i+1] - idx_offset[i],
                          step=1/f)
        sir = analytic_SIR(p, diameter/2, r_focus, c)
        append!(spr_analytic, map(t -> conv_sir(sir, pulse, t, c, pulse_length, 1e-8), t_samples))
        append!(sir_analytic, map(t -> sir(t), t_samples))
    end
    return spr_analytic, sir_analytic
end
