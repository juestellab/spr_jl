module SPR
    export AbstractTransducer,
           CylindricalTransducer,
           SphericalTransducer,
           FlatTransducer,
           dS, unsafe_position, total_surface, position, surface_samples,
           gaussian_pixel_model,
           spherical_pixel_model,
           soft_boundary_spherical_pixel_model,
           gaussian_pixel_model_deriv,
           Nshape,
           compute_spr,
           compute_spr_point,
           parse_spr_args,
           compute_spr_from_args,
           grid_from_parameters,
           save_spr,
           prefilter

    const default_computation_params = Dict(:atol => :auto,
                                            :rtol => 1e-4,
                                            :rtol_global => 1e-2,
                                            :threshold => 1e-5, 
                                            :maxevals => 1_000_000,
                                            :initdiv => 20)
    include("utils.jl")
    include("transducer.jl")
    include("pulse.jl")
    include("spr_core.jl")
    include("cli.jl")
    include("sir_prefilter.jl")
end
