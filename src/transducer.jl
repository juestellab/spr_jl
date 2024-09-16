using StaticArrays
import Base.iterate

abstract type AbstractTransducer{T<:AbstractFloat} end
corners(transducer::AbstractTransducer) = (transducer.min_corner, transducer.max_corner)
"""
    dS(transducer)
Measure of a transducer surface element
"""
dS(::AbstractTransducer) = error("Not Implemented")
"""
    total_surface(transducer)
Full surface area of the transducer
"""
total_surface(::AbstractTransducer) = error("Not Implemented")
unsafe_position(::AbstractTransducer, vararg...) = error("Not Implemented")


function position(transducer::AbstractTransducer, q1, q2)
    c1, c2 = corners(transducer)
    @assert all([q1, q2] .>= c1) && all([q1, q2] .<= c2)
    unsafe_position(transducer, q1, q2)
end

function surface_samples(transducer::AbstractTransducer, sampling)
    c1, c2 = corners(transducer)
    return [range(c1[i], c2[i], length=sampling[i]) for i in eachindex(c1)]
end

valid_fov_point(::AbstractTransducer, ::AbstractVector) = error("Not Implemented")


"""
Transducer object, that corresponds to a segment of a cylinder of some width and arc-length
"""
struct CylindricalTransducer{T} <: AbstractTransducer{T}
    radius::T
    min_corner::SVector{2,T}
    max_corner::SVector{2,T}
end
CylindricalTransducer(radius, min_corner::AbstractVector, max_corner::AbstractVector) = CylindricalTransducer(radius, SVector{2}(min_corner), SVector{2}(max_corner))

"""
    CylindricalTransducer(; radius::Number, arc_length::Number, width::Number)

Transducer with a cylindrical shape.

# Arguments
radius          Radius of the cylinder segment
arc_length      Arc length of the transducer
width           Width of the transducer
"""
CylindricalTransducer(; radius::Number, arc_length::Number, width::Number) = begin
    min_corner = SA[-width/2, -arc_length/(2*radius)]
    max_corner = -min_corner
    return CylindricalTransducer(radius, min_corner, max_corner)
end

@inline function unsafe_position(transducer::CylindricalTransducer, d, ϕ)
    r = transducer.radius
    return SA[d, r*sin(ϕ), r*(1.0-cos(ϕ))]
end

dS(transducer::CylindricalTransducer, _, _) = transducer.radius
total_surface(transducer::CylindricalTransducer) =
    *((transducer.max_corner - transducer.min_corner)...) * transducer.radius


struct SphericalTransducer{T} <: AbstractTransducer{T}
    radius::T
    min_corner::SVector{2,T}
    max_corner::SVector{2,T}
end

SphericalTransducer(radius::T, min_corner::AbstractVector, max_corner::AbstractVector) where {T} =
    SphericalTransducer(radius, SVector{2,T}(min_corner), SVector{2,T}(max_corner))
SphericalTransducer(; radius, diameter, throughhole_diameter = 0.) = 
    SphericalTransducer(radius, SA[0, asin(throughhole_diameter / (2 * radius))], SA[2π, asin(diameter / (2 * radius))])

@inline function unsafe_position(transducer::SphericalTransducer, ϕ, θ)
    r = transducer.radius
    sinθ, cosθ = sincos(θ)
    sinϕ, cosϕ = sincos(ϕ)
    return SA[r*sinϕ*sinθ, r*cosϕ*sinθ, r*(1.0-cosθ)]
end

dS(transducer::SphericalTransducer, _, θ) = transducer.radius^2 * sin(θ)
total_surface(transducer::SphericalTransducer) = 2π * transducer.radius^2 * (cos(transducer.min_corner[2]) - 0.5 * sin(transducer.min_corner[2])^2
                                                                            -cos(transducer.max_corner[2]) + 0.5 * sin(transducer.max_corner[2])^2)

"""
    valid_fov_point(transducer::SphericalTransducer, fov::AbstractVector)::Bool
Checks for a single point, whether it is suitable for the SPR computation.
It is checked if the point is below the lowest point of the transducer and if
it is within the cone spanned by the tangents at the transducer edges.
"""
function valid_fov_point(transducer::SphericalTransducer, fov::AbstractVector)::Bool
    ϕ = corners(transducer)[2][2]
    tangent_slope = cos(ϕ) / sin(ϕ)
    offset = transducer.radius * (1 - 1/cos(ϕ))

    below_transducer = fov[3] > transducer.radius * (1 - cos(ϕ))
    in_opening_cone = sqrt(fov[1]^2  + fov[2]^2) / (fov[3] - offset) < tangent_slope

    return below_transducer && in_opening_cone
end


struct FlatTransducer{T} <: AbstractTransducer{T}
    radius::Missing
    min_corner::SVector{2,T}
    max_corner::SVector{2,T}
end

FlatTransducer(missing, min_corner::AbstractVector, max_corner::AbstractVector) = FlatTransducer(missing, SVector{2}(min_corner), SVector{2}(max_corner))

FlatTransducer(;diameter::Number, throughhole_diameter::Number) = begin
    min_corner = SA[throughhole_diameter/2, 0]
    max_corner = SA[diameter/2, 2π]
    return FlatTransducer(missing, min_corner, max_corner)
end

@inline function unsafe_position(::FlatTransducer, d, ϕ)
    return SA[d*cos(ϕ), d*sin(ϕ), 0.0]
end

dS(::FlatTransducer, d, _) = d
total_surface(transducer::FlatTransducer) = 2π*(transducer.max_corner[1]^2 - transducer.min_corner[1]^2)


abstract type AbstractTransducerArray{T<:AbstractTransducer} end

struct CylindricalTransducerArray{T} <: AbstractTransducerArray{T}
    radius::AbstractFloat
    ϕ_positions::Union{AbstractFloat,AbstractArray}
    global_position::AbstractArray
    transducer_element::T
end

function iterate(T::CylindricalTransducerArray)
    (ϕ, s) = iterate(T.ϕ_positions)
    return (T.global_position .+ T.radius.*sincos(ϕ), s)
end

function iterate(T::CylindricalTransducerArray, state)
    iter = iterate(T.ϕ_positions, state)
    isnothing(iter) && return nothing
    (ϕ, s) = iter
    return (T.global_position .+ T.radius.*sincos(ϕ), s)
end
