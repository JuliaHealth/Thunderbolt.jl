# RefPoint (refdim = 0)
struct RefPoint <: AbstractRefShape{0} end
Ferrite.reference_vertices(::Type{RefPoint}) = (1,)
Ferrite.reference_edges(::Type{RefPoint}) = () # -
Ferrite.reference_faces(::Type{RefPoint}) = () # -

struct Point <: AbstractCell{RefPoint}
    nodes::NTuple{1, Int}
end
Point(i::Int) = Point((i,))
Ferrite.cell_to_vtkcell(::Type{Point}) = VTKCellTypes.VTK_VERTEX

struct PointInterpolation <: Ferrite.ScalarInterpolation{RefPoint, 0} end
Ferrite.getnbasefunctions(ip::PointInterpolation) = 1
Ferrite.adjust_dofs_during_distribution(::PointInterpolation) = false
