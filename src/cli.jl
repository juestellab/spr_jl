using ArgParse
import ArgParse.parse_item
using JSON
using StaticArrays

struct KeyWords
    kwargs::Dict{Symbol, Any}
end

function parse_item(::Type{KeyWords}, x::AbstractString)
    pairs = split.(split(x, ','), '=')
    return KeyWords(Dict((Symbol(k), parse(Float64, v)) for (k, v) in pairs))
end

struct ExtendingRange
    range::StepRangeLen
end

function parse_item(::Type{ExtendingRange}, x::AbstractString)
    start, step, stop = parse.(Float64, split(x, ':'))
    @assert stop >= start
    @assert step >= 0.0
    length = step > 0 ? Int(cld(stop - start, step)) + 1 : 1
    return ExtendingRange(range(start=start, length=length, step=step))
end

function parse_spr_args(device_keys)
    in_availible_devices(device) = device in device_keys

    s = ArgParseSettings("High-level script to facilitate parallel computation of SPRs")
    @add_arg_table! s begin
        "device_name"
            range_tester = in_availible_devices
            help = "Name of the device for which to compute the SPR.\n Choices: $(device_keys)"
        "--output_directory", "--od"
             help = "Directory into which to save the results of the computation."

        "--temporal_supersampling_factor", "--tsf"
            help = "Factor by which the SPR time resolution is larger than the device\"s DAQ time resolution. Default: 10"
            arg_type = Int
            default = 10
        "--relative_decay_cutoff_factor", "--rdcf"
            help = "Relative magnitude w.r.t. peak below which SPR values will be discarded. Default: 1e-6"
            arg_type = Float64
            default = 1e-6
        "--nthreads"
            arg_type = Int
            default = Sys.CPU_THREADS
    end

    add_arg_group!(s, "Cubature Options")
    @add_arg_table! s begin
        "--nquad_atol"
            help = "Absolute tolerance of Cubature. If 0, will be set to reasonable default"
            arg_type = Float64
            default = 0.
        "--nquad_rtol"
            help = "Relative tolerance of Cubature"
            arg_type = Float64
            default = 1e-6
        "--nquad_maxevals"
            arg_type = Int
            default = 2_000_000
        "--nquad_initdiv"
            arg_type = Int
            default = 20
    end

    add_arg_group!(s, "Coordinate System")
    @add_arg_table! s begin
        "polar"
            help = "Polar Coordiantes"
            action = :command

        "cartesian"
            help = "Cartesian Coordiantes"
            action = :command
    end
    @add_arg_table! s["polar"] begin
        "fov_radial_distance"
            arg_type = ExtendingRange
        "fov_azimuthal_angle"
            arg_type = ExtendingRange
        "fov_elevation"
            arg_type = ExtendingRange
    end
    @add_arg_table! s["cartesian"] begin
        "fov_x"
            arg_type = ExtendingRange
        "fov_y"
            arg_type = ExtendingRange
        "fov_z"
            arg_type = ExtendingRange
    end

    add_arg_group!(s, "Pulse Shape")
    for key in ["polar", "cartesian"]
        @add_arg_table! s[key] begin
            "n-shape"
                action = :command
            "gaussian_pulse"
                action = :command
        end
        @add_arg_table! s[key]["n-shape"] begin
            "pixel_model_type"
            "pixel_model_params"
                arg_type = KeyWords
        end
        @add_arg_table! s[key]["gaussian_pulse"] begin
            "--standard_deviation", "--sigma"
        end
    end

    return parse_args(s)
end

function compute_spr_from_args(args::Dict, devices)
    device = devices[args["device_name"]]
    f_sampling = device["daq"]["aquisition_frequency"] * args["temporal_supersampling_factor"]
    speed_of_sound = device["probe"]["coupling_medium"]["nominal_speed_of_sound"]
    
    cubature_opts = Dict(:atol => args["nquad_atol"], 
                         :rtol => args["nquad_rtol"],
                         :maxevals => args["nquad_maxevals"],
                         :initdiv => args["nquad_initdiv"],
                         :threshold => args["relative_decay_cutoff_factor"])

    ################################################################################ 
    # Field of View points
    ################################################################################ 
    if haskey(args, "polar")
        coords = "polar"
        fov_args = args["polar"]
        dist, angles, elevation = map(p -> p.range, [fov_args["fov_radial_distance"],
                                                     fov_args["fov_azimuthal_angle"],
                                                     fov_args["fov_elevation"]])

        fov_points = [SA[r * sind(ϕ), e, r * cosd(ϕ)] for r in dist, ϕ in angles, e in elevation]
    elseif haskey(args, "cartesian")
        coords = "cartesian"
        fov_args = args["cartesian"]
        X, Y, Z = map(p -> p.range, [fov_args["fov_x"], fov_args["fov_y"], fov_args["fov_z"]])
        fov_points = [SA[x, y, z] for x in X, y in Y, z in Z]
    end

    ################################################################################ 
    # Transducer and Pulse Function
    ################################################################################ 
    transducer = create_transducer(device["probe"]["transducer_array"]["transducer_properties"])

    if haskey(args[coords], "n-shape")
        pulse_params = args[coords]["n-shape"]
        pulse = Nshape(eval(:($(Symbol(pulse_params["pixel_model_type"]))(;$pulse_params["pixel_model_params"].kwargs...))))
    elseif haskey(args[coords], "gaussian_pulse")
        pulse_params = args[coords]["gaussian_pulse"]
        pulse = gaussian_pixel_model(pulse_params["standard_deviation"])
    end

    ################################################################################ 
    # Compute and Save
    ################################################################################ 
    res = compute_spr(fov_points, transducer, pulse,
                      f_sampling, speed_of_sound, cubature_opts)
    return res

end

function create_transducer(args)
    t = Symbol(args["type"])
    kwargs = Dict{Symbol,Any}()
    for (key, val) in args
        if key != "type"
            kwargs[Symbol(key)] = val
        end
    end
    transducer = eval(:($t(;$kwargs...)))
    return transducer
end

