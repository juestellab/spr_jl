using MAT
using HDF5
using SPR
using ProgressMeter
using StaticArrays
using Printf

using Distributed
nprocs() != 1 || addprocs(;exeflags="--project=.")
@everywhere using SPR

include("spherical_transducer_analytical.jl")

function load_parameters(file)
    vals = matopen(file) do f
        diameter = read(f, "r") * 2
        r_focus = read(f, "r_focus")

        points = read(f, "points")
        fov = [SA[points[:,i]...] for i in axes(points, 2)]

        t_start_sir = read(f, "time_start_SIR")
        t_length_sir = Int.(read(f, "length_SIR"))

        t_start_spr = read(f, "time_start_SPMR")
        t_length_spr = Int.(read(f, "length_SPMR"))
        return diameter, r_focus, fov, t_start_sir, t_length_sir, t_start_spr, t_length_spr
    end
    return vals
end


function compute_spr_julia(fov, time_samples, transducer, c, res, compute_params)
    pulse = Nshape(gaussian_pixel_model(res))
    idx_offset = [0, cumsum(length.(time_samples))...]
    
    vals, max_err, mse, compute_times = SPR.compute_spr_lowlevel(fov, time_samples, idx_offset, transducer, pulse, c, compute_params)
    vals_split = [vals[(1+idx_offset[i]):idx_offset[i+1]] for i=1:length(fov)]
    matwrite(@sprintf("results/julia_spr_rtol=%1.1e_initdiv=%s.mat",compute_params[:rtol], compute_params[:initdiv]),
             Dict("spr" => vals_split,
                  "max_error" => Array(max_err),
                  "mse" => Array(mse),
                  "compute_times" => Array(compute_times),
                  "points" => hcat(Array.(fov)...)
                  )
             )
end


function compute_spr_analytic(f_sampling_str, res, c)
    pulse = Nshape(gaussian_pixel_model(res))
    pulse_length = 6*res

    f_sampling = parse(Float64, f_sampling_str)
    diameter, r_focus, fov, t_start_sir, t_length_sir, t_start_spr, t_length_spr = load_parameters("FIELD_SPR_Sims/SIR_Field_comp_fs_$f_sampling_str.mat")
    sir_analytic = []
    spr_analytic = []
    @showprogress for (i, p) in enumerate(fov)
        order = (p[1] < 1.25e-2 && p[3] < 0.4e-2) ? 20 : 50
        factor = Int(f_sampling/40e6)
        t_sir = range(start=t_start_sir[i], length=t_length_sir[i]÷factor, step=1/40e6)
        t_spr = range(start=t_start_spr[i], length=t_length_spr[i]÷factor, step=1/40e6)
        sir = analytic_SIR(p, diameter/2, r_focus, c)
        push!(sir_analytic, map(t -> sir(t), t_sir))
        if p[1] == 0
            push!(spr_analytic, map(t -> analytic_SPR(p[3], diameter/2, r_focus, c, res)(t), t_spr))
        else
            push!(spr_analytic, map(t -> conv_sir(sir, pulse, t, c, pulse_length, 1e-10, order=order), t_spr))
        end
    end
    matwrite("results/analytic_sir_spr_f_sampling=$f_sampling_str.mat",
             Dict("f_sampling" => f_sampling,
                  "t_start_SIR" => t_start_sir,
                  "sir_analytic" => sir_analytic,
                  "t_start" => t_start_spr,
                  "spr_analytic" => spr_analytic,
                  "points" => hcat(Array.(fov)...)
                  )
             )
end


function main()
    c = 1500.0
    res = 50e-6

    diameter, r_focus, fov, _, _, t_start_spr, t_length_spr = load_parameters("FIELD_SPR_Sims/SIR_Field_comp_fs_40e6.mat")
    transducer = SphericalTransducer(diameter=diameter, radius=r_focus)
    t_spr = [range(start=t_start_spr[i], length=t_length_spr[i], step=1/40e6) for i in eachindex(t_start_spr)]


    for f_sampling_str in ["40e6", "80e6", "400e6", "1000e6", "5000e6", "10000e6"]
        compute_spr_analytic(f_sampling_str, res, c)
    end

    for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        compute_params = Dict(:atol => 2.9e-4 * 1e-4 * tol,
                              :rtol => tol,
                              :maxevals => 5_000_000,
                              :initdiv => 30
                              )
        compute_spr_julia(fov, t_spr, transducer, c, res, compute_params)
    end
end

    
res = main()
