############
# adapt.jl #
############
Adapt.@adapt_structure QuadratureValuesIterator
Adapt.@adapt_structure StaticQuadratureValues

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    ξs = Adapt.adapt(to, ntuple(i -> Adapt.adapt(to, cv.qr.points[i]), n_quadoints))
    FVT = typeof(fv)
    GMT = typeof(gm)
    Nqp = length(weights)
    T = eltype(weights)
    dim = ξs |> first |> length
    return StaticCellValues{FVT, GMT, Nqp, T, dim}(fv, gm, weights, ξs)
end
