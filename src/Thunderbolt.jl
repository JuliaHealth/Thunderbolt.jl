module Thunderbolt

using TimerOutputs: @timeit_debug

import SciMLLogging: Standard, AbstractVerbosityPreset, @SciMLMessage

# These two must be imported *before* anything that pulls in Tensors (which `FerriteOperators` does,
# via Ferrite). Loading Tensors first makes parts of Symbolics'/ModelingToolkit's precompiled images
# fail validation, so they are recompiled inside their `__init__` — measured at ~4.5 s of extra load
# time per Julia process (12.1 s -> 7.6 s), of which `Symbolics.__init__` alone goes 0.1 ms -> 1464 ms
# at 100% recompilation.
#
# This is a *cache* effect, not method invalidation: `@snoop_invalidations using Tensors` reports zero
# invalidated MethodInstances. It therefore depends on the current Tensors/SIMD/Symbolics combination
# and can regress silently on an upgrade — the load-time check in CLAUDE.md is what catches that.
import DynamicQuantities

import FerriteOperators:
    FerriteOperators,
    SequentialCPUDevice,
    PolyesterDevice,
    duplicate_for_device,
    InternalVariableHandler,
    AbstractAssemblyStrategy,
    AbstractCPUDevice,
    AssemblyStrategy,
    default_strategy,
    FullAssembly,
    MatrixFreeAction,
    BlockRowAssembly,
    StandardOperatorSpecification,
    SequentialScheduling,
    ColoredScheduling,
    AbstractGPUDevice,
    AbstractNonlinearIntegrator,
    AbstractCondensedNonlinearIntegrator,
    AbstractNonlinearOperator,
    QuadratureRuleCollection,
    getquadraturerule,
    setup_element_cache,
    AbstractVolumetricElementCache,
    AbstractSurfaceElementCache,
    EmptySurfaceElementCache,
    EmptyVolumetricElementCache,
    FacetItemDomain,
    update_linearization!,
    evaluate!,
    assemble_cell!,
    assemble_algebraic!,
    reinit_values!,
    provides_analytic,
    has_internal_state,
    get_number_of_internal_dofs_per_element,
    setup_internal_variable_handler,
    condense_cell!,
    condense_internal!,
    CondensationReport,
    ResidualRequest,
    JacobianRequest,
    JacobianResidualRequest,
    WeightedJacobianRequest,
    JacobianKind,
    JacobianResidualKind,
    WeightedJacobianKind,
    CellArgs,
    FacetArgs,
    with_states,
    TimeIntegrationContext,
    evaluation_time,
    stage_scaling,
    AffineRate,
    InternalSource,
    assemble_weighted_jacobian!,
    internal_variable_offset,
    AbstractBilinearIntegrator,
    AbstractLinearIntegrator,
    assemble_facet!,
    functional_value_type,
    value_type

import FerriteInterfaceElements:
    InterfaceCellInterpolation, InterfaceCellValues, InterfaceCell, getdetJdV_average

import FerriteOperators:
    AbstractBilinearOperator,
    AbstractLinearOperator,
    BilinearFerriteOperator,
    LinearFerriteOperator,
    LinearNullOperator,
    NullOperator,
    setup_operator,
    setup_evaluation_operator,
    update_operator!,
    setup_qvector,
    get_range_for_cell,
    evaluate_quadrature!,
    get_dof_handler,
    get_strategy,
    get_subdomain_caches

import Unrolled: @unroll
import FastBroadcast: @..

using UnPack: @unpack # TODO remove this package
using Reexport: @reexport
import LinearAlgebra: mul!
import Polyester: @batch
using SparseMatricesCSR, LinearAlgebra
using OrderedCollections: OrderedDict, OrderedSet
using BlockArrays, SparseArrays, StaticArrays

using JLD2: jldopen
import WriteVTK
import ReadVTK

import OrdinaryDiffEqOperatorSplitting as OS
import OrdinaryDiffEqOperatorSplitting: GenericSplitFunction
export OS, GenericSplitFunction
function solution_size(gsf::GenericSplitFunction)
    alldofs = Set{Int}()
    for solution_indices in gsf.solution_indices
        union!(alldofs, solution_indices)
    end
    return length(alldofs)
end

# Children report positions local to themselves and the split rebases them. Solution indices are relative
# to the parent at every level, so this composes through arbitrarily nested splits -- electrophysiology
# and mechanics split apart, then each split again -- without a special case.
function solution_variables(gsf::GenericSplitFunction)
    vars = SolutionVariable[]
    for i = 1:OS.num_operators(gsf)
        indices = OS.get_solution_indices(gsf, i)
        for v in solution_variables(OS.get_operator(gsf, i))
            push!(vars, translate(v, indices))
        end
    end
    return merge_and_check_unique(vars)
end

@reexport using Ferrite
import Ferrite:
    AbstractDofHandler,
    AbstractGrid,
    AbstractRefShape,
    AbstractCell,
    get_grid,
    get_coordinate_eltype,
    addfacetset!
import Ferrite: vertices, edges, facets, faces, sortedge, sortface
import Ferrite: get_coordinate_type, getspatialdim

import Preferences

import Logging: Logging, LogLevel, @info, @logmsg

import SymbolicIndexingInterface
import SciMLBase
@reexport import SciMLBase: init, solve, solve!, step!
@reexport import SciMLIterators: TimeChoiceIterator
using RecursiveArrayTools: recursivecopy!, recursivecopy
import DiffEqBase#: AbstractDiffEqFunction, AbstractDEProblem
import OrdinaryDiffEqCore#: OrdinaryDiffEqCore
import OrdinaryDiffEqCore:
    DummyController, DummyControllerCache, default_controller, setup_controller_cache
import LinearSolve
using LinearSolve: LinearAliasSpecifier

import ConcreteStructs: @concrete

using Base: @kwdef
import Base: *, +, -

import ForwardDiff

# Accelerator support libraries
using Adapt: @adapt_structure, Adapt

include("mesh/meshes.jl")

include("utils.jl")

include("ferrite-addons/transfer_operators.jl")
include("ferrite-addons/point.jl")


# Note that some modules below have an "interface.jl" but this one has only a "common.jl".
# This is simply because there is no modeling interface, but just individual physics modules and couplers.
include("modeling/common.jl")

include("modeling/microstructure.jl")

include("modeling/electrophysiology.jl")
include("modeling/solid_mechanics.jl")
include("modeling/fluid_mechanics.jl")

include("modeling/multiphysics.jl")

include("modeling/solution_variables.jl")

include("modeling/functions.jl")
include("modeling/problems.jl")

# Diagnostics dispatch on the function layer, so they come after it.
include("modeling/solid/diagnostics.jl")

include("gpu/gpu_utils.jl")

include("discretization/interface.jl")
include("discretization/fem.jl")
include("discretization/operator.jl")

include("solver/logging.jl")
include("solver/interface.jl")
include("solver/linear.jl")
include("solver/nonlinear.jl")
include("solver/time_integration.jl")

include("modeling/electrophysiology/ecg.jl")

include("ferrite-addons/io.jl")

include("disambiguation.jl")

# TODO where to put these?
include("modeling/rsafdq2022.jl")
include("discretization/rsafdq-operator.jl")

# Last: the workload solves have to see every model, discretization and solver above.
include("precompile.jl")

# The `MTKModels` circuit definitions live in `ThunderboltMTKExt`; reach them via `mtk_models()`.

# TODO put exports into the individual submodules above!
export
    # Angle between two directions about an axis, left hand rule
    compute_relative_rotation,
    # Long axis of a ventricular geometry
    LongAxisInfo,
    compute_long_axis,
    fit_basal_plane,
    compute_principal_axis

export
    # Devices
    SequentialCPUDevice,
    PolyesterDevice,
    # Coefficients
    ConstantCoefficient,
    FieldCoefficient,
    AnalyticalCoefficient,
    FieldCoefficient,
    SpectralTensorCoefficient,
    SpatiallyHomogeneousDataField,
    setup_coefficient_cache,
    evaluate_coefficient,
    # Collections
    LagrangeCollection,
    DiscontinuousLagrangeCollection,
    getinterpolation,
    QuadratureRuleCollection,
    NodalQuadratureRuleCollection,
    getquadraturerule,
    CellValueCollection,
    getcellvalues,
    FacetValueCollection,
    getfacetvalues,
    # Mesh generators
    generate_mesh,
    generate_open_ring_mesh,
    generate_ring_mesh,
    generate_quadratic_ring_mesh,
    generate_quadratic_open_ring_mesh,
    generate_ideal_lv_mesh,
    generate_ideal_lh_mesh,
    # Mesh utilities
    hexahedralize,
    separate_chamber_surfaces,
    to_mesh,
    # Generic models
    TransientDiffusionModel,
    InterfaceDiffusionModel,
    AffineODEFunction,
    # Named access to the solution vector
    default_initial_condition!,
    create_initial_condition,
    solution_variables,
    solution_variable,
    solution_variable_names,
    solution_indices,
    getvariable,
    setvariable!,
    FieldVariable,
    LocalStateVariable,
    GlobalVariable,
    CellIndexCoordinateSystem,
    # Local API
    PointwiseODEProblem,
    PointwiseODEFunction,
    # Mechanics
    QuasiStaticModel,
    QuasiStaticProblem,
    QuasiStaticFunction,
    ElastodynamicsModel,
    ElastodynamicsProblem,
    ElastodynamicsFunction,
    PK1Model,
    PrestressedMechanicalModel,
    # Passive material models
    NullEnergyModel,
    NullCompressionPenalty,
    SimpleCompressionPenalty,
    HartmannNeffCompressionPenalty1,
    HartmannNeffCompressionPenalty2,
    HartmannNeffCompressionPenalty3,
    TransverseIsotopicNeoHookeanModel,
    HolzapfelOgden2009Model,
    LinYinPassiveModel,
    LinYinActiveModel,
    HumphreyStrumpfYinModel,
    BioNeoHookean,
    Guccione1991PassiveModel,
    Guccione1993ActiveModel,
    LinearSpringModel,
    SimpleActiveSpring,
    # Contraction model
    AsRateIndependent,
    CaDrivenInternalSarcomereModel,
    ConstantStretchModel,
    PelceSunLangeveld1995Model,
    RDQ20MFModel,
    # Active model
    ActiveMaterialAdapter,
    GMKActiveDeformationGradientModel,
    GMKIncompressibleActiveDeformationGradientModel,
    RLRSQActiveDeformationGradientModel,
    SimpleActiveStress,
    PiersantiActiveStress,
    # Electrophysiology
    MonodomainModel,
    # ParabolicParabolicBidomainModel,
    # ParabolicEllipticBidomainModel,
    NoStimulationProtocol,
    TransmembraneStimulationProtocol,
    AnalyticalTransmembraneStimulationProtocol,
    ReactionDiffusionSplit,
    gating_symbols,
    gate_coefficients,
    gating_indices,
    # Circuit
    RSAFDQ2022LumpedCicuitModel,
    MTKLumpedCicuitModel,
    # FSI
    RSAFDQ2022Model,
    RSAFDQ2022Split,
    LumpedFluidSolidCoupler,
    ChamberVolumeCoupling,
    Pressure3D0DVolumeCoupler,
    ChamberVolumeFunctional,
    chamber_volume,
    # Microstructure
    AnisotropicPlanarMicrostructureModel,
    AnisotropicPlanarMicrostructure,
    OrthotropicMicrostructureModel,
    OrthotropicMicrostructure,
    TransverselyIsotropicMicrostructureModel,
    TransverselyIsotropicMicrostructure,
    ODB25LTMicrostructureParameters,
    NoMicrostructureModel,
    create_microstructure_model,
    # Coordinate system
    LVCoordinateSystem,
    LVCoordinate,
    BiVCoordinateSystem,
    BiVCoordinate,
    CartesianCoordinateSystem,
    LocalCoordinateAxes,
    setup_coordinate_axes_cache,
    evaluate_coordinate_axes,
    LVAxes,
    compute_lv_axes,
    compute_lv_coordinate_system,
    compute_midmyocardial_section_coordinate_system,
    apicobasal_from_laplace,
    getcoordinateinterpolation,
    getrotationalinterpolation,
    vtk_coordinate_system,
    # Discretization
    semidiscretize,
    FiniteElementDiscretization,
    # Solver
    SchurComplementLinearSolver,
    KrylovMGSolver,
    AbstractMGPrecon,
    PMGPrecon,
    GMGPrecon,
    ChainedMGPrecon,
    EisenstatWalkerForcing,
    NewtonRaphsonSolver,
    MultiLevelNewtonRaphsonSolver,
    HomotopyPathSolver,
    BackwardEulerSolver,
    NewmarkSolver,
    PIDController,
    # Super-time-stepping families
    RKC1,
    RKL1,
    RKG1,
    # Exponential multirate super-time-stepping for a reaction-diffusion split
    EMRKC,
    # Convergence driven step size control, usable with any solver answering `contraction_rate_cache`
    Deuflhard2004DiscreteContinuationController,
    Deuflhard2004_B_DiscreteContinuationControllerVariant,
    ForwardEulerCellSolver,
    AdaptiveForwardEulerSubstepper,
    # Integrator
    # Utils
    QuadraturePoint,
    QuadratureIterator,
    load_carp_grid,
    load_voom2_grid,
    load_mfem_grid,
    solution_size,
    velocity,
    acceleration,
    # IO
    ParaViewWriter,
    JLD2Writer,
    store_timestep!,
    store_timestep_celldata!,
    store_timestep_field!,
    # store_coefficient!,
    finalize_timestep!,
    finalize!,
    # Mechanical PDEs
    GeneralizedHillModel,
    ActiveStressModel,
    ExtendedHillModel,
    #  BCs
    NormalSpringBC,
    PressureFieldBC,
    BendingSpringBC,
    RobinBC,
    ConstantPressureBC,
    #  Kinematic diagnostics
    DeformationMonitor,
    #  Viscous (dashpot) BCs
    ViscousRobinBC,
    ViscousNormalSpringBC
end
