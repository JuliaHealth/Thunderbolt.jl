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

Ferrite.getnbasefunctions(ip::Lagrange{RefPoint, 0}) = 1
Ferrite.adjust_dofs_during_distribution(::Lagrange{RefPoint, 0}) = false
