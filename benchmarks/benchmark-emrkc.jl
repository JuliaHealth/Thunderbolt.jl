# emRKC against the production reaction-diffusion split, host and device: on an ep01-shaped monodomain
# sheet for two cell models, and on an ideal left ventricle of ~1e6 hexahedra carrying a full
# orthotropic fiber architecture. What it costs to buy a simulated millisecond with an explicit
# stabilized scheme, next to `LieTrotterGodunov(BackwardEulerSolver + Jacobi-preconditioned
# KrylovJL_CG, ForwardEulerCellSolver)`.
#
# MATCHED ACCURACY, NOT MATCHED STEP SIZE. Comparing the two at one step size would measure nothing:
# the step size each method can afford is a property of the method, and that is the whole claim under
# test. So each picks its own, and they are compared where they deliver the same error.
#
#  * Reference: each method against ITSELF at `DTREF`, in Float64 on the host. Deliberately not a
#    shared reference -- emRKC lumps the mass matrix and the backward Euler stage does not, so the two
#    integrate different semidiscretizations of the same monodomain problem. That gap is printed
#    below; it is of the order of the accuracy band itself, and a shared reference would lay it under
#    both error curves as a floor that no step size could get beneath. Each reference's own error is
#    estimated by its distance to the same method at `2 DTREF`, printed beside it. The backward Euler
#    reference tightens its conjugate gradient well past the tutorial's setting, because otherwise the
#    linear solve's own error -- which accumulates with the step count -- is what the reference
#    measures. The timed arms keep the tutorial's tolerances, so that floor shows up where it belongs:
#    in the error curve of the configuration a user would actually run.
#  * Step size: for each method, the largest in `SWEEP` whose final-time φₘ relative error stays
#    inside `BAND`. The sweep runs in the precision and on the hardware the host arm is timed in, so
#    the calibration *is* the host arm's validation.
#  * Validation: every timed arm is checked against the reference at its own step size before it is
#    timed -- the device arms repeat the check in their own precision (Float32, or Float64 where noted
#    below) on the device. A silently wrong kernel must not produce a headline number.
#  * Timing: minimum over `NPASS` passes of `NSTEPS` steps. Device arms first step for
#    `WARMUP_SECONDS` without interruption, because this card idles at 300 MHz and a cold arm measures
#    the clock ramp instead of the kernel; the clocks are read back per arm and printed, so the reader
#    can see the card was hot. Timing starts from the initial condition having had `WARMUP_SECONDS`
#    worth of real steps to develop -- for FHN that is a spiral wave, for PCG2019 see below.
#  * Composition: for the splitting arms the per step time is split over the two children through the
#    same calls `OrdinaryDiffEqOperatorSplitting`'s own `_perform_step!` makes, which separates the
#    linear solve from the pointwise reaction. emRKC has no linear solve to separate out; what is
#    reported in its place is the stage structure `(s, m)` that stands in for one.
#  * Knife edge -- and on FHN it has now flipped, which is the single most important thing to read off
#    the tables below. Roughly half of the reported speedup is the step size each method lands on, and
#    that half sits at the accuracy band's edge rather than in its middle. FHN's splitting arm used to
#    miss the band at Δt = 0.40 and fall back to 0.20; that is where the 3.38x host / 3.04x device
#    ratios this file reported before came from. The Jacobi preconditioner lowers the linear solve's
#    own error floor at the same tolerances, Δt = 0.40 now lands *inside* the band (0.00983 against
#    0.01, a 1.7% margin), both methods take the same step, and FHN's ratio falls to 1.47x / 1.86x --
#    per-step cost with no step-size component left in it. A 2% shift in that one error puts it back.
#    So: FHN's ratio is a knife edge and should be quoted as one. PCG2019's is not -- its splitting arm
#    NaNs outright at Δt >= 0.20, so 0.10 against emRKC's 0.20 is a hard factor of two. The other half
#    is per step and is not a knife edge either: emRKC takes no linear solve, which is 79-93% of a
#    splitting step below.
#
# MODELS: `ENV["EMRKC_MODELS"]`, comma-separated, defaults to `"FHN,PCG2019,LV"` -- they run in one
# invocation (one Julia startup, one CUDA warmup) back to back, and a model whose arms error out is
# caught and reported without stopping the others. `BAND`, `NSTEPS`, `NPASS`, `WARMUP_SECONDS` are
# shared, unchanged discipline across all three; `N`, `TEND`, `DTREF`, `SWEEP` apply to the two sheet
# configs, which select their own step size, and the LV overrides them (see the LV-PCG2019 block).
# The sheet configs are `ModelConfig`s in `MODEL_CONFIGS` below; the LV is an `LVConfig` with its own
# `run_model` method, because its geometry, its coefficients and above all its validation discipline
# are different enough that sharing one struct would mean a field that means nothing to the other two.
#  * FHN: 2.5mm x 2.5mm, the ep01 tutorial's own dimensionless diffusion tensor, an excite/refractory
#    box initial condition that develops into a sustained spiral.
#  * PCG2019: 10mm x 10mm, κ/(Cₘχ) = 0.4 mm²/ms -- the physiological value already validated for this
#    model in `test/integration/test_emrkc_electrophysiology.jl` -- and that same test's planar S1
#    front (left 12% of the domain excited to 20mV, the rest at the model's own resting default). That
#    test reports the front taking ~15-20ms to cross this 10mm domain (mesh-resolution independent, so
#    unchanged at this benchmark's N=512), so at TEND=25ms, and through most of the timed passes'
#    warmup, the tissue is genuinely mixed -- upstroke, plateau and still-resting side by side -- which
#    is what both the accuracy check and the per-step cost need to see; once it has crossed the domain
#    settles into a shared plateau (a real AP's plateau lasts ~200-300ms, past this benchmark's own
#    horizon either way). `EMRKC(gates = :all)` is spelled out explicitly: PCG2019's six gates
#    (`h,m,f,s,xs,xr`) are exactly what the exponential treatment is for, and `gates = ()` would
#    silently run past them.
#  * LV: an ideal left ventricle, 1026600 hexahedra / 1081605 nodes / 7571235 states, PCG2019 again,
#    with a genuinely orthotropic conductivity built from a Streeter helix (+60 endo to -60 epi) over
#    the LV coordinate system. Its geometry, conductivities, protocol and -- the part that did not go
#    to plan -- its step sizes are all set out in the LV-PCG2019 block below.
#
# THREADING: host arms run 2 threads -- the capped profile this file is meant to be run under is
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p CPUWeight=25 env
# JULIA_NUM_THREADS=2 julia -t2 --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl`.
# `ThreadedSparseMatrixCSR`'s SpMV is threaded across those 2 host threads; the pointwise reaction/gate
# sweeps are not threaded (single-threaded on host, one CUDA thread per dof on device). Uncapped/-t8
# arms are not measured here. This applies to both models.
#
# DEVICE PRECISION: every device arm here is `Float32`, and genuinely so. That was not true of
# PCG2019 until `src/modeling/cells/pcg2019.jl` stopped hardcoding bare `1.0`/`2.0`/`-1.0` literals in
# `_pcg2019_sigmoid` and the `h`-gate's `τ_h`: those widened every sigmoid and every
# `gate_coefficients` call to `Float64` regardless of the struct's `T`, so a "Float32" device arm ran
# its hottest loop in double precision on a card built for single -- `@code_typed` on the `Float32`
# instantiation showed 77 `Float64`-mentioning statements across `cell_rhs_fast!`, `cell_rhs_slow!`
# and `gate_coefficients`, and now shows none. The `Float64` path is bit-identical across that change
# (`T(literal)` at `T = Float64` is exact, and `test/test_gating_protocol.jl` pins it against a
# verbatim pre-factoring reference), so the PCG2019 *host* numbers below are comparable to the ones
# this file reported before it; the device numbers are not, and are remeasured.
#
# PRECONDITIONING: the splitting arms' conjugate gradient carries a Jacobi (diagonal) preconditioner,
# host and device, through `KrylovJL_CG`'s own `precs` seam -- see `JacobiPrecon` below for why it has
# to fill itself lazily rather than at `init`. The unpreconditioned iteration count is measured once
# per configuration and printed beside the preconditioned one, because the linear solve is 76-95% of a
# splitting step and what the preconditioner removes from that is the single largest lever on the
# emRKC-vs-splitting ratio. emRKC has no linear solve and is untouched by it.
#
# EXPECTATION (PCG2019, stated before measuring): its reaction is expensive relative to FHN's -- 6
# exponential gates and stiff kinetics (`τ_m = 0.12ms`, λ ≈ -8.3/ms) against FHN's one linear gate --
# which dilutes the relative weight of "no linear solve", emRKC's whole advantage over the split. Both
# arms' step sizes are expected reaction-accuracy-limited to ~0.2-0.4ms per prior 0D single-cell
# probes, similar in scale to each other, which further narrows what the step size itself buys (the
# "knife edge" bullet above). FHN's measured 2.38x host / 2.68x device ratios are a ceiling here, not
# an expectation -- PCG2019 is measured below, not assumed.
#
# MEASURED OUTCOME vs. that expectation: the step sizes landed where expected (0.2ms/0.1ms, inside the
# 0.2-0.4ms band), and PCG2019's ratio is now the larger of the two -- 2.93x host and 5.49x device
# against FHN's 1.47x/1.86x. Most of that gap is FHN's knife edge flipping rather than PCG2019 gaining;
# PCG2019's own ratio came *down*, from 3.46x/8.45x, when the preconditioner went in. The reaction
# dilution is real but small (the reaction is 19% of a PCG2019 host splitting step against FHN's 21%).
# What actually separates the two models is conditioning: PCG2019's physiological κ = 0.4mm²/ms makes
# the backward Euler system far worse conditioned than FHN's dimensionless one -- 114.8 unpreconditioned
# CG iterations per step against FHN's 8.0 -- which is why the linear solve is 81-93% of a PCG2019
# splitting step, why splitting's host step is 10x FHN's, and why Jacobi is worth 1.93x of iteration
# count here and only 1.6x there. The device ratio still exceeds the host one (5.49x > 2.93x), the
# opposite of FHN's pattern, but no longer because of a hidden Float64: both arms are genuinely Float32
# now, and what remains is that a 45-iteration CG chain of dependent SpMVs and reductions maps onto this
# card worse than emRKC's 23 independent stabilized stages do.
#
# This is a benchmark, not a CI gate: the numbers are reported as measured, whichever way they fall.
#
# CUDA is a weak dependency, so this runs in the GPU test environment rather than the package one:
# `julia --project=test/gpu benchmarks/benchmark-emrkc.jl`.
#
# Requires a memory-capped cgroup (enforced by `_assert_memory_capped()` below; `BENCHMARK_UNCAPPED=1`
# overrides it) -- an uncapped run's GC sizes itself against the whole machine, not the 8G it is meant
# to run in. Canonical invocation:
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 julia -t2
# --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl`.
#
# RESULTS (most recent run, capped 2-thread host profile, RTX 2080; N=512, 512x512 mesh for the two
# sheet models; every device arm Float32; every splitting arm Jacobi-preconditioned):
#
# == FHN (526338 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.40      0.02260      44.2     0.05651       none        -   s=1 m=2      host
# host splitting         0.40      0.03330      30.0     0.08325        79%      5.0         -      host
# device emRKC           0.40      0.00099    1011.9     0.00247       none        -   s=1 m=2 1905/6800
# device splitting       0.40      0.00184     542.9     0.00461        81%      5.0         - 1905/6800
# host->device: emRKC 22.87x, splitting 18.08x  |  emRKC vs splitting: 1.47x host, 1.86x device
# CG iterations/step at Δt=0.40, host: 8.0 unpreconditioned, 5.0 with Jacobi
# validation (rel err, band=0.01): host emRKC 0.00968, host splitting 0.00983,
#   device emRKC 0.00968, device splitting 0.00983 -- all in band
#
# == PCG2019 (1842183 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.20      0.23544       4.2     1.17718       none        -  s=1 m=23      host
# host splitting         0.10      0.34435       2.9     3.44351        81%     59.4         -      host
# device emRKC           0.20      0.00405     247.2     0.02023       none        -  s=1 m=23 1905/6800
# device splitting       0.10      0.01111      90.0     0.11109        93%     45.2         - 1890/6800
# host->device: emRKC 58.20x, splitting 31.00x  |  emRKC vs splitting: 2.93x host, 5.49x device
# CG iterations/step at Δt=0.10, host: 114.8 unpreconditioned, 59.4 with Jacobi
# validation (rel err, band=0.01): host emRKC 0.00157, host splitting 0.00342,
#   device emRKC 0.00158, device splitting 0.00337 -- all in band
# splitting sweep: Δt=0.05 err=0.00108, Δt=0.10 err=0.00342 (selected), Δt≥0.20 NaN (blew up --
#   the stiff gate's FE substep limit, as expected; the selected Δt is safely below it)
# emRKC sweep: Δt=0.05/0.10/0.20 in band (err 0.00099/0.00139/0.00157), Δt=0.40 err=0.020 (out);
#   0.20ms selected, inside the 0.2-0.4ms window the 0D probes anticipated
#
# == LV-PCG2019 (1031400 hexahedra, 1065057 nodes, 7455399 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.0500    2.65125       0.4    53.02502       none        -  s=1 m=28      host
# host splitting         0.0068    1.22868       0.8   180.68839        84%     47.0         -      host
# device emRKC           0.0500    0.02844      35.2     0.56883       none        -  s=1 m=28 1905/6800
# device splitting       0.0068    0.04946      20.2     7.27423        94%     46.6         - 1890/6800
# host->device: emRKC 93.22x, splitting 24.84x  |  emRKC vs splitting: 3.41x host, 12.79x device
# CG iterations/step at Δt=0.0068, host: 83.5 unpreconditioned, 47.0 with Jacobi (1.78x)
# host-vs-device agreement over 5 steps, both Float32: emRKC 3.38e-6, splitting 1.13e-6
# memory: host peak 3.85 GiB of the 8 GiB cgroup (52% headroom), device 0.72 GiB of 7.60 (90%)
#
# ITEM 1 -- APEX MESH QUALITY, measured rather than assumed. At matched ~1.03M-element resolution,
# `generate_ideal_lv_mesh` (the fan+`hexahedralize` this file uses) gives true 12-edge-per-hexahedron
# min/median/max 3.18/134.8/188.7 µm (median/min = 42). `generate_ideal_lv_mesh_hex`'s all-hex O-grid
# cap, at the SAME target dimensions and a matched total element count (nc=348, nr=20, nl=125; nc has
# to be divisible by 4; `septum_flatness`/`axis_ratio`/`eccentricity` zeroed to match the fan's forced
# rotational symmetry, isolating the apex-meshing comparison from an unrelated shape change), gives
# 1.41/132.3/221.7 µm (median/min = 94): its minimum edge is SMALLER, not larger, and its median/min
# spread is more than double the fan's. A second probe at the fan's own pre-`hexahedralize` resolution
# (176, 10, 73) locates the defect inside the O-grid CORE specifically: core-only edges run
# 4.75/7.63/197.7 µm, a 26x spread WITHIN the core alone, so this is not one unlucky corner cell but a
# structural property of the core's transfinite-interpolation mapping at this resolution. Per the
# decision rule (switch only if the O-grid's minimum edge is materially better), the answer is: stay
# on `generate_ideal_lv_mesh`. The alternative does not fix the apex, it relocates a same-order sliver
# into the core and gives it a worse median/min spread. Mesh quality goes back to Dennis as a
# `generate_ideal_lv_mesh_hex` core-mapping improvement task, not a config change here. (Note: earlier
# runs of this file quoted a 160 µm median from an all-pairwise-node-distance proxy that folds in face
# and body diagonals; the true-edge figures above are a methodology correction, not a mesh change --
# both are measured on the OLD geometry for this comparison.)
#
# ITEM 2 -- CHAMBER RE-PROPORTIONING (Dennis's ruling, binding): keep h ≈ 150 µm AND full transmural
# resolution (wall 6 mm ≈ 40 elements) and reach ~1e6 elements by shrinking the OTHER dimensions
# instead. `generate_ideal_lv_mesh`'s four radii are no longer tied to one scale: `inner_radius`/
# `outer_radius` fix the wall at exactly `outer_radius = inner_radius + 6` mm; `apex_inner`/
# `apex_outer` set the long axis independently -- breaking the generator's default 0.7:1.0:1.3:1.5
# proportions on purpose. Final dimensions: endocardial equatorial radius 3.112 mm (cavity, shrunk
# from 6.916 mm), epicardial equatorial radius 9.112 mm (wall 6.000 mm exactly), apex_inner/apex_outer
# 5.780/6.669 mm, apex-base length (outer surface) 8.73 mm (shrunk from 19.39 mm). base = (191, 20, 33)
# before `hexahedralize`, coarse_base = (96, 10, 17). Result: 1,031,400 hexahedra, 1,065,057 nodes,
# 7,455,399 states (7 states/node, PCG2019), 40 transmural elements at h = 150 µm exactly; median edge
# (true-edge measure) 126.0 µm, wall volume 1.501 mL, mean cell volume 1.455e-3 mm³ (h_eff 113.3 µm --
# the same "polar topology packs elements toward the apex" shortfall as before). The chamber is a
# thick-walled, small, non-physiological ventricle by design -- the accepted trade for keeping h and
# the transmural count. Memory never became the binding constraint here either (52% / 90% headroom on
# the timed mesh, see MEMORY below): the arithmetic (this time shrinking the OTHER dimensions) is what
# got it to ~1e6.
#
# DIFFUSION is genuinely orthotropic, not transversely isotropic: `SpectralTensorCoefficient` over an
# `OrthotropicMicrostructureModel` built by `create_simple_microstructure_model` on the
# `compute_lv_coordinate_system` frame, with a Streeter helix running +60° at the endocardium to -60°
# at the epicardium. σ = (0.13342, 0.02674, 0.00859) mS/mm in the (fiber, sheetlet, normal) frame,
# with Cₘ = 0.01 µF/mm² and χ = 140 /mm giving D = (0.0953, 0.0191, 0.0061) mm²/ms. Sources, and their
# class: the fiber value is the harmonic mean of Clerc's measured intra- and extracellular
# longitudinal conductivities, which is the monodomain reduction the Niederer et al. N-version
# benchmark uses and the same number `ep04_geselowitz-ecg.jl` already carries; the cross-fiber pair
# splits that benchmark's single transverse value (0.0176) by the squared ratios of the orthotropic
# conduction velocities Caldwell et al. (2009) measured in ventricular tissue (0.67 : 0.30 : 0.17
# m/s), and brackets it. Measured tissue data throughout, not a fit to this benchmark.
# That the microstructure actually reaches the assembly is checked rather than assumed, on the coarse
# mesh, by running the identical problem with a trace-matched isotropic tensor: the two disagree by
# 0.169 after 20 steps (0.214 on the previous mesh -- different geometry integrates differently; both
# are comfortably above the round-off floor). A microstructure that never arrived would agree to
# round-off.
#
# DEVICE PATH for the LV is host-assembled and mirrored, not device-assembled. The field-backed
# `OrthotropicMicrostructureModel` stores its f/s/n vectors in host `ElementwiseData` with no adapt
# rule, so it cannot cross the `KernelAbstractionsDevice` assembly seam; the host strategy assembles
# and `MirroredBilinearOperator` uploads the nonzeros, which is a supported configuration and is what
# the sheet device arms use too. Assembly is setup-only at fixed Δt, so this costs the timed arms
# nothing. What it means for the numbers: the device arms solve with exactly the host's orthotropic
# matrix, and the host-vs-device agreement above (3.4e-6 / 1.1e-6) is what says the upload is intact.
#
# PROTOCOL: an apex S1 written as an initial condition -- the apical 12% of the long axis raised to
# 20 mV, the rest at PCG2019's resting default -- over a 15 ms window, no full beat. At 15 ms the
# tissue is genuinely mixed, which is what the per-step cost needs to see: 57.1% of it above -40 mV
# with φₘ ∈ [-85, 21] mV under emRKC, 77.1% and [-88, 23] mV under splitting.
#
# WHAT THE LV ADDS to the sheet picture: emRKC's advantage is larger here than anywhere else in this
# file -- 3.41x host and 12.79x device -- for the same structural reason as before: both methods pay
# for the apical slivers, but they pay differently. emRKC absorbs them into its stage count (m=28 at
# the certified Δt=0.05, one extra SpMV per stage), while the splitting arm pays through BOTH a step
# size about 7x smaller than emRKC's AND a linear solve that is 84-94% of its step. The device column
# compounds it further: a 47-iteration CG chain of dependent SpMVs and reductions maps onto this card
# far worse than emRKC's 28 independent stabilized stages, hence 12.79x against 3.41x on the host.
#
# WHAT MOVED SINCE THE PREVIOUS MESH (item 2's re-proportioning) -- every LV number above is fresh,
# not comparable step-for-step to the old (174,10,73)-based mesh, but the DIRECTION of each change is
# informative:
#  * emRKC got CHEAPER: m dropped from 32 to 28 stages at the same Δt=0.05 (still certified directly,
#    rel err 0.00326 vs 0.00176 before -- both comfortably in band), and s/step fell 12% (2.65 vs
#    3.01) even though the new mesh has slightly MORE cells (1,031,400 vs 1,026,600) -- the smaller,
#    thicker-walled proportions apparently lower the diffusive spectral radius ρS overall, not only at
#    the apex.
#  * splitting's step got SMALLER and stayed uncertified: Δt=0.0068 (Richardson-estimated) against the
#    old mesh's Δt=0.01 -- see RE-CERTIFICATION below. Fewer, cheaper CG solves (47.0 vs 63.4
#    preconditioned iterations) were not enough to offset the smaller step: splitting got 10% MORE
#    expensive per simulated ms (180.7 vs 164.3 s/sim-ms, host), not less.
#  * Net effect: emRKC's advantage over splitting GREW -- 3.41x host (was 2.73x), 12.79x device (was
#    10.37x) -- driven entirely by splitting's smaller, less certain step size, not by any change to
#    either method's per-step machinery.
#  * host-vs-device agreement improved incidentally (3.4e-6/1.1e-6 vs 1.2e-5/4.4e-6); both are already
#    far below anything that would matter, so this is noise, not a result.
#
# RE-CERTIFICATION (mandated by item 2's mesh change): emRKC's carried Δt=0.05 is CERTIFIED DIRECTLY
# on the new mesh -- the full sweep was repeated: 0.20/0.10/0.05/0.025 give rel err
# 0.1294/0.03402/0.003255/0.0005165, so 0.05 is again the largest in-band value. splitting remains
# UNCERTIFIED: its DTREF=0.025 reference has NOT converged (own error 0.0369, worse than the old
# mesh's 0.0241), and the code's own linear Richardson estimate (E(Δt) ≈ 1.48·Δt) gives Δt ≲ 0.0068,
# which is what is timed. One level deeper was tried, per this slice's instruction: DTREF=0.0125 gives
# own error 0.0262, STILL unconverged -- only a 29% improvement for a 2x refinement (ratio 0.71,
# apparent local convergence order ≈ 0.49), well short of the ~50% a first-order method should shed.
# That sub-linear behavior is itself informative: it suggests the COARSE MESH's own spatial error may
# be becoming a Δt-independent floor at this refinement, not purely a time-stepping issue, so a still
# finer DTREF may not converge either without also refining the coarse mesh -- the previous run's
# "~3 hours, was not spent" estimate for DTREF ≲ 0.003 may now be optimistic. Worth flagging rather
# than hiding: at the carried Δt=0.0068 the arm's error against the (unconverged) DTREF=0.025
# reference is 0.0528, LARGER than the old Δt=0.01's 0.0342 -- non-monotonic in Δt, which is a symptom
# of an unconverged/biased reference (moving the arm closer to the truth moves it further from a
# reference that is not at the truth), not a contradiction. splitting's step size and its "uncertified"
# label both stand.
#
# ITEM 3 -- MEMORY ATTRIBUTION (host, measured after setup of all four arms on this mesh, before
# timing; `Base.summarysize` per retained object class; full per-arm table kept outside this file).
# The mesh+DofHandler (257.6 MiB) and the three-field orthotropic microstructure (582.3 MiB,
# `ElementwiseData` stored per-cell-LOCAL-node -- duplicated at cell boundaries, neither a shared
# nodal array nor per-quadrature-point) are SHARED once across all four arms, not duplicated per arm
# (852.1 MiB total shared graph). Per arm beyond that: host splitting alone retains THREE full-size
# sparse matrices at once -- mass M and diffusion K, each a 437.7 MiB Float64 `SparseMatrixCSC`
# (FerriteOperators' own assembly dtype, independent of the Float32 arms), PLUS the actual Float32
# `ThreadedSparseMatrixCSR` combination M-ΔtK (218.8 MiB) the CG solve uses -- M and K are never freed
# once that combination is built. Each device arm additionally carries a HOST-resident staging copy of
# its own K (and, for splitting, M too) for `MirroredBilinearOperator`'s upload -- 437.7 MiB plus a
# 107.4 MiB Float32 staging buffer per matrix -- while the device-native combined system matrix, CG
# workspace and Jacobi preconditioner are genuinely GPU-resident and cost the host under 4 KiB each.
# Holding all four arms' setup simultaneously (a deliberate stress test for this attribution, NOT what
# the timed run does) peaks at host RSS 7.29 GiB / VmHWM 7.48 GiB / cgroup memory.peak 6.89 GiB of the
# 8 GiB cap -- 6-9% headroom, uncomfortably close. The REAL timed run releases each arm via `GC.gc()`
# before building the next (see `run_arm`), and its own measured peak is the 3.85 GiB / 52% headroom
# quoted above -- comfortably inside the 25% floor. Memory levers, not applied here (Dennis's call):
# Float32 host-arm M/K instead of Float64 would roughly halve their combined ~875 MiB per arm;
# dropping M/K once the combined system matrix is assembled would save that same ~875 MiB per
# host-splitting-shaped arm; releasing the device arms' host-side mirror staging copies (437.7-1090.2
# MiB per arm) after the one-time upload would recover up to ~1.6 GiB across both device arms; a
# Float32 microstructure would roughly halve its 582.3 MiB (shared, so a one-time saving).

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using StaticArrays
using Printf

import Ferrite
import SparseArrays: nonzeros
import SparseMatricesCSR: getrowptr, getcolval, getnzval
import Thunderbolt: SciMLBase, ThreadedSparseMatrixCSR, create_simple_microstructure_model
import OrdinaryDiffEqOperatorSplitting:
    advance_solution_by!, forward_sync_subintegrator!, backward_sync_subintegrator!

const N              = 512          # dofs/node-count shared across models, the size at which the
                                     # solve dominates -- independent of the physical domain size L
const TEND           = 25.0         # ms, the first EP tutorial's own visualization window
const DTREF          = 0.025        # ms, reference step size
const BAND           = 1.0e-2       # final-time φₘ relative error a step size has to stay inside
const SWEEP          = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
const NSTEPS         = 25
const NPASS          = 3
const WARMUP_SECONDS = 1.5
const CuCSR          = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

####################################
## Jacobi preconditioning
####################################

"""
The reciprocal main diagonal of the backward Euler operator `M - Δt K`, as a left preconditioner for
the conjugate gradient.

Filled on first use rather than at `init`, because `LinearSolve.init` calls the `precs` callback while
the system matrix is still the freshly allocated all-zero sparsity pattern -- and the affine backward
Euler path then fills that same matrix object in place through `nonzeros(A)` and never reassigns
`cache.A`, which is what would otherwise mark the preconditioner stale. A preconditioner built at
`init` would therefore be the preconditioner of the zero matrix, forever. Holding on to `A` and
reading its diagonal on the first `ldiv!` sidesteps both halves of that: by the time the conjugate
gradient asks for a preconditioner application, the matrix it is preconditioning is assembled.

Every arm in this file runs at a fixed `Δt`, so `A` is assembled once and never changes again; the
one-shot fill is valid for the life of the arm. A varying `Δt` would have to invalidate `ready`.
"""
mutable struct JacobiPrecon{MatType, VecType}
    A::MatType
    inv_diag::VecType
    ready::Bool
end

function JacobiPrecon(A)
    d = similar(nonzeros(A), size(A, 1))
    fill!(d, one(eltype(d)))
    return JacobiPrecon(A, d, false)
end

# Neither `ThreadedSparseMatrixCSR` nor `CuSparseMatrixCSR` has a `diag` method or a scalar
# `getindex`, so each row is walked for its own column index -- on the host directly, on the device
# one thread per row. Anything else (the dense/CSC matrices the coordinate system solves hand over)
# takes the generic `diag` path.
function _fill_inv_diag!(d, A::ThreadedSparseMatrixCSR)
    rowptr, colval, nzval = getrowptr(A), getcolval(A), getnzval(A)
    @inbounds for i in eachindex(d)
        v = zero(eltype(d))
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            colval[k] == i && (v = nzval[k])
        end
        d[i] = iszero(v) ? one(v) : inv(v)
    end
    return d
end

function _inv_diag_kernel!(d, rowPtr, colVal, nzVal)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i ≤ length(d)
        v = zero(eltype(d))
        for k in rowPtr[i]:(rowPtr[i + 1] - 1)
            colVal[k] == i && (v = nzVal[k])
        end
        d[i] = iszero(v) ? one(v) : inv(v)
    end
    return nothing
end

function _fill_inv_diag!(d::CuVector, A::CUDA.CUSPARSE.CuSparseMatrixCSR)
    threads = 256
    CUDA.@cuda threads = threads blocks = cld(length(d), threads) _inv_diag_kernel!(
        d, A.rowPtr, A.colVal, A.nzVal,
    )
    return d
end

function _fill_inv_diag!(d, A)
    v = diag(A)
    d .= ifelse.(iszero.(v), one(eltype(d)), inv.(v))
    return d
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPrecon, x::AbstractVector)
    if !P.ready
        _fill_inv_diag!(P.inv_diag, P.A)
        P.ready = true
    end
    y .= P.inv_diag .* x
    return y
end
LinearAlgebra.ldiv!(P::JacobiPrecon, x::AbstractVector) = ldiv!(x, P, x)

# `KrylovJL`'s own contract: `(A, p) -> (Pl, Pr)`. CG takes left/centered preconditioning only, so the
# right slot stays the identity.
jacobi_precs(A, p = nothing) = (JacobiPrecon(A), LinearAlgebra.I)

"""
Krylov iterations taken by the most recent backward Euler solve. `stats` is the workspace's own
object, reset per solve, so the count is read out per step rather than kept.
"""
cg_iters(child) = child.cache.stage.linear_solver.cacheval.stats.niter

"Mean conjugate gradient iterations per step over `n` real steps. The integrator is advanced."
function mean_cg_iters!(integrator, n = NSTEPS)
    total = 0
    for _ = 1:n
        step!(integrator)
        total += cg_iters(integrator.child_subintegrators[1])
    end
    return total / n
end

####################################
## Problem
####################################

"One sheet model's mesh extent, monodomain coefficients, ionic model constructor and initial condition."
struct ModelConfig
    name::String
    ion::Function                    # (::Type{T}) -> ionic model instance
    Cₘ::Float64
    χ::Float64
    κ::SymmetricTensor{2, 2, Float64}
    L::Float64                       # domain side
    u0!::Function                    # (u₀, form, ::Type{T}) -> u₀
end

function fhn_u0!(u₀, form, ::Type{T}) where {T}
    setvariable!(u₀, form, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? one(T) : zero(T)
    end
    setvariable!(u₀, form, :s) do x
        x[2] ≥ 1.25 ? T(0.1) : zero(T)
    end
    return u₀
end

const PCG2019_L = 10.0

"""
A planar S1 activation front: the left 12% of the domain driven to a real excited potential, the rest
at the model's own resting default (`create_initial_condition` already wrote it) -- the same protocol
`test/integration/test_emrkc_electrophysiology.jl` validates for this model, scaled to this
benchmark's domain. A first attempt used a cross-field excite/refractory-block IC (bottom-left
quadrant excited, top half's `h` gate forced to 0) to chase a sustained spiral the way FHN's own IC
does; that discontinuous `h` jump kept emRKC's self-referenced reference error (`DTREF` vs `2DTREF`)
above `BAND` regardless of step size -- not a step-size-selection failure but an unconverged
reference, an artifact of the synthetic discontinuity rather than a property of emRKC or PCG2019
worth reporting. The planar front has no such discontinuity and converges cleanly.
"""
function pcg2019_u0!(u₀, form, ::Type{T}) where {T}
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ T(0.12PCG2019_L) ? T(20.0) : T(-85.0)
    end
    return u₀
end

const FHN_CONFIG = ModelConfig(
    "FHN", T -> Thunderbolt.ParametrizedFHNModel{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5)), 2.5, fhn_u0!,
)
const PCG2019_CONFIG = ModelConfig(
    "PCG2019", T -> Thunderbolt.ParametrizedPCG2019Model{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((0.4, 0.0, 0.4)), PCG2019_L, pcg2019_u0!,
)
const AVAILABLE_MODELS = Dict{String, Any}(
    "FHN" => FHN_CONFIG,
    "PCG2019" => PCG2019_CONFIG,
    # Defined further down, once the LV section has introduced `LVConfig`.
)
const MODEL_CONFIGS = String[strip(m) for m in split(get(ENV, "EMRKC_MODELS", "FHN,PCG2019,LV"), ",")]

function ep01_form(::Type{T}, cfg::ModelConfig) where {T}
    mesh = generate_mesh(Quadrilateral, (N, N), Vec{2}((0.0, 0.0)), Vec{2}((cfg.L, cfg.L)))
    model = MonodomainModel(
        ConstantCoefficient(cfg.Cₘ),
        ConstantCoefficient(cfg.χ),
        ConstantCoefficient(cfg.κ),
        NoStimulationProtocol(),
        cfg.ion(T),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

function ep01_u0(form, ::Type{T}, cfg::ModelConfig) where {T}
    u₀ = create_initial_condition(form, T)
    cfg.u0!(u₀, form, T)
    return u₀
end

# The two algorithms under test, as a downstream user would spell them. The linear solver tolerances
# are the first EP tutorial's. `gates = :all` is EMRKC's own default; spelled out here so the choice is
# visible in the file rather than implicit (see the PCG2019 header note).
emrkc(::Type{VT}, ::Type{MT}) where {VT, MT} =
    EMRKC(solution_vector_type = VT, system_matrix_type = MT, gates = :all)

function splitting(
    ::Type{VT}, ::Type{MT}; atol = 1.0e-6, rtol = 1.0e-5, jacobi = true,
) where {VT, MT}
    T = eltype(VT)
    cg = jacobi ?
        KrylovJL_CG(atol = T(atol), rtol = T(rtol), precs = jacobi_precs) :
        KrylovJL_CG(atol = T(atol), rtol = T(rtol))
    return LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type = VT,
            system_matrix_type   = MT,
            inner_solver         = cg,
        ),
        ForwardEulerCellSolver(solution_vector_type = VT),
    ))
end

# `init` takes the initial condition as the integrator's own state, so every arm gets a copy: the
# host and device initial conditions are shared across arms and would otherwise be consumed by the
# first one to run.
build(form, u0, alg, Δt, tend) =
    init(OperatorSplittingProblem(form, copy(u0), (zero(Δt), tend)), alg; dt = Δt, verbose = false)

function solve_to_end(form, u0, alg, Δt, tend = TEND)
    integrator = build(form, u0, alg, Δt, oftype(Δt, tend))
    solve!(integrator)
    return integrator
end

"`n` real steps from the initial condition. The LV arms validate this way instead of over the whole
window: at a million dofs a full-window host solve per arm costs more than the timing it guards, and
what the check is for -- that the device reproduces the host -- is already visible after a few steps."
function solve_n_steps(form, u0, alg, Δt, n)
    integrator = build(form, u0, alg, Δt, oftype(Δt, 1.0e5))
    for _ = 1:n
        step!(integrator)
    end
    return integrator
end

sync(::Vector) = nothing
sync(::CuVector) = CUDA.synchronize()

relerr(a, b) = norm(Vector(a) .- Vector(b)) / norm(Vector(b))

####################################
## Reference and step size selection
####################################

"""
The Float64 host reference of one method, and the estimate of its own error: the distance to the same
method at twice the step size, which for a first order scheme bounds the reference's error up to the
factor two between them.
"""
function reference(form, u0, alg, φₘ, tend = TEND)
    coarse = getvariable(solve_to_end(form, u0, alg, 2DTREF, tend).u, φₘ)
    fine   = getvariable(solve_to_end(form, u0, alg, DTREF, tend).u, φₘ)
    return copy(fine), relerr(coarse, fine)
end

"""
The largest step size in `SWEEP` whose final state stays inside `BAND` of `φ_ref`, together with the
error it lands at. Errors are printed for the whole sweep: where the band is crossed is as much of
the result as which step size wins.
"""
function select_dt(form, u0, alg, φₘ, φ_ref, label)
    best = nothing
    for Δt in SWEEP
        # In the solution vector's precision: `(t, Δt)` reach the reaction kernel, and a Float64 step
        # size against Float32 storage runs the hot loop in double precision.
        integrator = solve_to_end(form, u0, alg, eltype(u0)(Δt))
        φ = getvariable(integrator.u, φₘ)
        ok = integrator.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
        err = ok ? relerr(φ, φ_ref) : NaN
        @printf("    %-8s Δt = %5.2f   rel err = %-10.4g %s\n", label, Δt, err,
                ok ? (err ≤ BAND ? "in band" : "") : "FAILED")
        ok && err ≤ BAND && (best = (Δt, err))
    end
    best === nothing && error("$label: no step size in SWEEP reaches the $BAND band.")
    return best
end

####################################
## Timing
####################################

function gpu_clocks()
    out = read(
        `nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits`, String,
    )
    sm, mem = parse.(Int, strip.(split(first(split(strip(out), '\n')), ',')))
    return sm, mem
end

"""
Step without interruption for `WARMUP_SECONDS` and read the clocks back *while still stepping*. On a
card that idles at 300 MHz a short warmup measures the ramp rather than the kernel; the returned
clocks are what says this one did not. They are sampled mid-flight and not after the loop, because
the card starts dropping back within tens of milliseconds of going idle -- which is less than one
`nvidia-smi` query takes.
"""
function prewarm!(integrator, on_device)
    t0 = time_ns()
    clocks = (0, 0)
    sampled = false
    while (time_ns() - t0) / 1.0e9 < WARMUP_SECONDS
        step!(integrator)
        if on_device && !sampled && (time_ns() - t0) / 1.0e9 > WARMUP_SECONDS / 2
            clocks = gpu_clocks()
            sampled = true
        end
    end
    sync(integrator.u)
    return clocks
end

"Minimum seconds per step over `NPASS` passes of `NSTEPS` steps."
function measure!(integrator)
    best = Inf
    for _ = 1:NPASS
        t0 = time_ns()
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
        best = min(best, (time_ns() - t0) / 1.0e9 / NSTEPS)
    end
    return best
end

"""
Seconds per step spent in each child of a splitting step, through the same three calls
`OrdinaryDiffEqOperatorSplitting`'s `_perform_step!` makes for it. Replicating the loop rather than
instrumenting it keeps this honest about what the parent step actually does; the integrator is
advanced by real steps and is left usable.

Also returns the mean conjugate gradient iterations of the first child, read out of the same steps
rather than from a second pass: every extra step advances the Float32 clock these arms run on, and
the splitting integrator's parent/child time synchronization is what pays for that drift.
"""
function child_composition!(integrator, Δt)
    times = zeros(length(integrator.child_subintegrators))
    iters = 0
    for _ = 1:NSTEPS
        for (i, child) in enumerate(integrator.child_subintegrators)
            idxs   = integrator.child_solution_indices[i]
            syncer = integrator.child_synchronizers[i]
            forward_sync_subintegrator!(integrator, child, idxs, syncer)
            t0 = time_ns()
            advance_solution_by!(integrator, child, Δt)
            sync(integrator.u)
            times[i] += (time_ns() - t0) / 1.0e9
            i == 1 && (iters += cg_iters(child))
            backward_sync_subintegrator!(integrator, child, idxs, syncer)
        end
    end
    return times ./ NSTEPS, iters / NSTEPS
end

####################################
## Arms
####################################

struct Arm
    label::String
    method::String
    Δt::Float64
    seconds_per_step::Float64
    validation::Float64
    solve_fraction::Float64   # share of the step in the linear solve; NaN where there is none
    cg_iters::Float64         # mean conjugate gradient iterations per step; NaN where there is none
    stages::String
    clocks::Tuple{Int, Int}
end

"""
Validate the arm, then time it. Returns the arm and the validated final `φₘ` on the host, so that a
host arm can serve as the reference its own device counterpart is checked against.

`steps` validates over that many steps from the initial condition instead of over the whole window;
`φ_ref === nothing` validates finiteness only, for an arm whose step size is certified elsewhere.
"""
function run_arm(
    label, method, form, u0, alg, Δt, φₘ, φ_ref, on_device, has_solve;
    tend = TEND, steps = nothing,
)
    # Validate before timing: the trajectory this arm produces, at the step size it was given.
    checked = steps === nothing ? solve_to_end(form, u0, alg, Δt, tend) :
              solve_n_steps(form, u0, alg, Δt, steps)
    φ = getvariable(Vector(checked.u), φₘ)
    all(isfinite, φ) || error("$label: validation produced a non-finite solution")
    err = φ_ref === nothing ? NaN : relerr(φ, φ_ref)
    stages = if has_solve
        ""
    else
        s, _, m = Thunderbolt._emrkc_step_sizing(checked.alg, Δt, checked.cache.ρS, checked.cache.ρF)
        "s=$s m=$m"
    end

    # Time from `WARMUP_SECONDS` of real steps past the initial condition, not from the initial
    # condition itself: `tend` here only has to outlast the warmup and the passes.
    timed = build(form, copy(u0), alg, Δt, oftype(Δt, 1.0e5))
    clocks = prewarm!(timed, on_device)
    seconds = measure!(timed)

    fraction, iters = NaN, NaN
    if has_solve
        parts, iters = child_composition!(timed, Δt)
        fraction = parts[1] / sum(parts)
    end
    arm = Arm(label, method, Δt, seconds, err, fraction, iters, stages, clocks)
    GC.gc()
    on_device && CUDA.reclaim()
    return arm, φ
end

function report(arms)
    println("\n", "="^128)
    @printf("%-22s %-7s %9s %9s %11s %10s %9s %9s %9s\n",
            "arm", "Δt/ms", "s/step", "steps/s", "s / sim ms", "lin solve", "cg its",
            "stages", "clocks")
    println("-"^128)
    for a in arms
        @printf("%-22s %-7.4f %9.5f %9.1f %11.5f %10s %9s %9s %9s\n",
                a.label, a.Δt, a.seconds_per_step, 1 / a.seconds_per_step,
                a.seconds_per_step / a.Δt,
                isnan(a.solve_fraction) ? "none" : @sprintf("%.0f%%", 100a.solve_fraction),
                isnan(a.cg_iters) ? "-" : @sprintf("%.1f", a.cg_iters),
                isempty(a.stages) ? "-" : a.stages,
                a.clocks == (0, 0) ? "host" : @sprintf("%d/%d", a.clocks[1], a.clocks[2]))
    end
    println("-"^128)
    for (host, device) in (("host emRKC", "device emRKC"), ("host splitting", "device splitting"))
        h = findfirst(a -> a.label == host, arms)
        d = findfirst(a -> a.label == device, arms)
        h === nothing || d === nothing || @printf(
            "  %-18s host -> device: %.2fx per simulated ms\n", arms[h].method,
            (arms[h].seconds_per_step / arms[h].Δt) / (arms[d].seconds_per_step / arms[d].Δt),
        )
    end
    for (a, b) in (("host emRKC", "host splitting"), ("device emRKC", "device splitting"))
        i = findfirst(x -> x.label == a, arms)
        j = findfirst(x -> x.label == b, arms)
        i === nothing || j === nothing || @printf(
            "  %-18s emRKC vs splitting: %.2fx per simulated ms\n",
            split(a)[1], (arms[j].seconds_per_step / arms[j].Δt) / (arms[i].seconds_per_step / arms[i].Δt),
        )
    end
    println("\nvalidation (φₘ relative error against this arm's reference):")
    for a in arms
        if isnan(a.validation)
            @printf("  %-22s finite; step size set on the coarse mesh, see the header block\n", a.label)
        else
            @printf("  %-22s %.6g %s\n", a.label, a.validation,
                    a.validation ≤ 2BAND ? "" : "  <-- OUT OF BAND")
        end
    end
end

####################################
## Ideal left ventricle
####################################

"""
The LV arm's geometry and step sizes. Unlike the sheet configs this one does not select its own step
size: it carries the sheet's, and certifies them on `coarse_base` (see `run_model(::LVConfig)`).
"""
struct LVConfig
    name::String
    base::NTuple{3, Int}          # circumferential, transmural, longitudinal, BEFORE hexahedralization
    coarse_base::NTuple{3, Int}
    Δt_emrkc::Float64
    Δt_split::Float64
end

# `generate_ideal_lv_mesh` emits a wedge fan over the apex, so the mesh is built at 2h and
# `hexahedralize`d: that both halves h and makes every cell a hexahedron, while keeping the fan
# variant's apex. `generate_ideal_lv_mesh_hex`'s O-grid cap was measured as an alternative (see the
# ITEM 1 block below) and rejected: its apex is not merely coarser, it is a worse sliver than the
# fan's, so the fan stays and the mesh-quality item is a generator-improvement task, not a config change.
#
# The chamber does NOT carry the generator's default proportions (wall/inner radius 0.3/0.7, long
# axis 1.3/1.5) at one uniform scale -- see the ITEM 2 block for why that combination cannot hit
# both h ≈ 150 µm and a 6 mm wall at ~1e6 elements. `inner_radius`/`outer_radius` fix the wall at
# exactly 6 mm; `apex_inner`/`apex_outer` set the long axis independently.
const LV_INNER_RADIUS = 3.1122   # mm, endocardial equatorial radius -- shrunk, non-physiological
const LV_OUTER_RADIUS = LV_INNER_RADIUS + 6.0   # mm, wall fixed at 6 mm (40 elements transmural)
const LV_APEX_INNER   = 5.7798   # mm
const LV_APEX_OUTER   = 6.6690   # mm, sets the long axis: apex-base length = LV_APEX_OUTER*(1-cospi(0.6))
const LV_Z_APEX  = LV_APEX_OUTER
const LV_Z_BASE  = LV_APEX_OUTER * cospi(0.6)
const LV_STIM_Z  = LV_Z_APEX - 0.12(LV_Z_APEX - LV_Z_BASE)
const LV_TEND    = 15.0          # ms; the front transits the apical wall and starts apicobasal
const LV_STEPS   = 5             # host-vs-device agreement steps on the timed mesh

# Monodomain conductivities, mS/mm, in the (fiber, sheetlet, normal) frame. The fiber value is the
# harmonic mean of Clerc's intra- and extracellular longitudinal conductivities, 0.17·0.62/(0.17+0.62)
# -- the monodomain reduction used by the Niederer et al. N-version benchmark, and the same number
# `ep04_geselowitz-ecg.jl` already carries. That benchmark is transversely isotropic (σ_s = σ_n =
# 0.0176); this one needs three distinct eigenvalues, so the cross-fiber pair is split by the squared
# ratios of the orthotropic conduction velocities Caldwell et al. (2009) measured in ventricular
# tissue, 0.67 : 0.30 : 0.17 m/s, which bracket that transverse value from either side. Source class:
# measured tissue conductivities and measured orthotropic conduction velocities, not a fit.
const LV_σ  = SVector(0.13342, 0.02674, 0.00859)
const LV_Cₘ = 0.01               # µF/mm²
const LV_χ  = 140.0              # 1/mm ; D = σ/(Cₘχ) = (0.0953, 0.0191, 0.0061) mm²/ms

host_rss_gib() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 1024^3
gpu_used_gib() = (CUDA.total_memory() - CUDA.free_memory()) / 1024^3

function lv_geometry(base)
    mesh = hexahedralize(generate_ideal_lv_mesh(
        base...;
        inner_radius = LV_INNER_RADIUS, outer_radius = LV_OUTER_RADIUS,
        apex_inner = LV_APEX_INNER, apex_outer = LV_APEX_OUTER, longitudinal_upper = 0.2,
    ))
    cs = compute_lv_coordinate_system(mesh)
    microstructure = create_simple_microstructure_model(
        cs, LagrangeCollection{1}()^3;
        endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0),
    )
    return mesh, microstructure
end

"""
`MonodomainModel`'s first two positional arguments are `χ, Cₘ` by its own field order, which every
call site in the repository spells the other way round; the two only ever enter as the product `Cₘχ`
in `κ/(Cₘχ)`, so the disagreement is invisible until someone gives them different values -- as this
does. Struct order it is.
"""
function lv_form(::Type{T}, mesh, microstructure; κ = nothing) where {T}
    model = MonodomainModel(
        ConstantCoefficient(LV_χ),
        ConstantCoefficient(LV_Cₘ),
        κ === nothing ? SpectralTensorCoefficient(microstructure, ConstantCoefficient(LV_σ)) : κ,
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedPCG2019Model{T}(),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

"An apex S1 stimulus written as an initial condition -- the apical 12% of the long axis raised above
threshold, the rest at the cell model's resting default. The same protocol shape the sheet PCG2019
config uses, and for the same reason: no applied-current amplitude to tune against excitability."
function lv_u0(form, ::Type{T}) where {T}
    u₀ = create_initial_condition(form, T)
    setvariable!(u₀, form, :φₘ) do x
        x[3] ≥ LV_STIM_Z ? T(20.0) : T(-85.0)
    end
    return u₀
end

"Fraction of the tissue above -40 mV, and the range of φₘ: what says the timed state is a front
mid-flight rather than uniformly resting or uniformly plateaued."
function activation_state(φ)
    v = Vector(φ)
    return count(>(-40.0f0), v) / length(v), minimum(v), maximum(v)
end

function run_model(cfg::LVConfig)
    println("\n", "#"^128)
    println("# ", cfg.name, "  (ideal LV, wall ", LV_OUTER_RADIUS - LV_INNER_RADIUS,
            " mm, PCG2019, gates = :all, device Float32)")
    println("#"^128)

    ############ the coarse mesh: what certifies the step sizes ############
    # The certification is a property of the physics and the step size, not of the timed mesh, so
    # `EMRKC_LV_COARSE=0` skips it when re-timing at step sizes a previous run already certified.
    # The default runs it.
    get(ENV, "EMRKC_LV_COARSE", "1") == "1" && lv_certify_step_sizes(cfg)

    ############ the timed mesh ############
    return lv_time_arms(cfg)
end

function lv_certify_step_sizes(cfg::LVConfig)
    println("\n-- coarse spot-check mesh --")
    t0 = time()
    cmesh, cms = lv_geometry(cfg.coarse_base)
    @printf("  %d hexahedra, %d nodes, setup %.0f s, host RSS %.2f GiB\n",
            Ferrite.getncells(cmesh.grid), Ferrite.getnnodes(cmesh.grid), time() - t0, host_rss_gib())

    cform64 = lv_form(Float64, cmesh, cms)
    cu64    = lv_u0(cform64, Float64)
    cform32 = lv_form(Float32, cmesh, cms)
    cu32    = lv_u0(cform32, Float32)
    φc64, φc32 = solution_variable(cform64, :φₘ), solution_variable(cform32, :φₘ)

    # Orthotropy is live, not merely configured: the same problem with a trace-matched isotropic
    # tensor has to produce a different solution. If the microstructure never reached the assembly
    # these two agree to round-off.
    σ_iso = sum(LV_σ) / 3
    ciso = lv_form(Float32, cmesh, cms; κ = ConstantCoefficient(
        SymmetricTensor{2, 3, Float64}((σ_iso, 0.0, 0.0, σ_iso, 0.0, σ_iso))))
    alg32 = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    φ_ortho = getvariable(solve_n_steps(cform32, cu32, alg32, Float32(cfg.Δt_emrkc), 20).u, φc32)
    φ_iso   = getvariable(solve_n_steps(ciso, lv_u0(ciso, Float32), alg32, Float32(cfg.Δt_emrkc), 20).u,
                          solution_variable(ciso, :φₘ))
    anisotropy = relerr(φ_ortho, φ_iso)
    @printf("  orthotropic vs trace-matched isotropic after 20 steps: rel diff %.4g %s\n",
            anisotropy, anisotropy > 1.0e-3 ? "(microstructure is live)" : "<-- SUSPECT")
    ciso = φ_iso = φ_ortho = nothing
    GC.gc()

    println("\n  self-referenced convergence at the carried step sizes (reference Float64, arms Float32):")
    cpu64_emrkc = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    cpu64_split = splitting(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64};
                            atol = 1.0e-12, rtol = 1.0e-10)
    cpu32_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    for (name, Δt, ref_alg, arm_alg) in (
        ("emRKC", cfg.Δt_emrkc, cpu64_emrkc, alg32),
        ("splitting", cfg.Δt_split, cpu64_split, cpu32_split),
    )
        φ_ref, ref_err = reference(cform64, cu64, ref_alg, φc64, LV_TEND)
        got = solve_to_end(cform32, cu32, arm_alg, Float32(Δt), Float32(LV_TEND))
        φ = getvariable(got.u, φc32)
        ok = got.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
        err = ok ? relerr(φ, φ_ref) : NaN
        frac, lo, hi = ok ? activation_state(φ) : (NaN, NaN, NaN)
        @printf("    %-10s Δt = %5.3f  rel err = %-9.4g %-8s\n",
                name, Δt, err, ok ? (err ≤ BAND ? "in band" : "OUT OF BAND") : "FAILED")
        # `ref_err` is `relerr(u(2·DTREF), u(DTREF))`, which for a first order method is also the
        # reference's own error. Once that reaches `BAND` the reference cannot certify anything -- the
        # arm is then being compared against something no more accurate than itself -- so say so rather
        # than print an error that looks like a verdict. It doubles as a Richardson estimate of the
        # step size that would be needed: E(Δt) ≈ (ref_err/DTREF)·Δt.
        if ref_err ≥ BAND
            @printf("               reference UNCONVERGED (own error %.3g ≥ band): the number above is\n",
                    ref_err)
            @printf("               not a verdict. Richardson: E(Δt) ≈ %.3g·Δt, so Δt ≲ %.4f for the band.\n",
                    ref_err / DTREF, BAND * DTREF / ref_err)
        else
            @printf("               reference's own error %.3g\n", ref_err)
        end
        @printf("               t = %.0f ms: %.1f%% of tissue above -40 mV, φₘ ∈ [%.1f, %.1f] mV\n",
                LV_TEND, 100frac, lo, hi)
    end
    cmesh = cms = cform64 = cform32 = cu64 = cu32 = nothing
    GC.gc()
    @printf("  host RSS after releasing the coarse mesh: %.2f GiB\n", host_rss_gib())
    return nothing
end

function lv_time_arms(cfg::LVConfig)
    println("\n-- timed mesh --")
    t0 = time()
    mesh, ms = lv_geometry(cfg.base)
    ncells, nnodes = Ferrite.getncells(mesh.grid), Ferrite.getnnodes(mesh.grid)
    form = lv_form(Float32, mesh, ms)
    u32  = lv_u0(form, Float32)
    φₘ32 = solution_variable(form, :φₘ)
    @printf("  %d hexahedra, %d nodes, %d states, setup %.0f s\n",
            ncells, nnodes, Thunderbolt.solution_size(form), time() - t0)

    cpu_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    gpu_emrkc = emrkc(CuVector{Float32}, CuCSR)
    gpu_split = splitting(CuVector{Float32}, CuCSR)
    ugpu = CuVector(u32)

    println("\nCG iterations per step, host, Δt = ", cfg.Δt_split, " ms:")
    plain = build(form, u32,
                  splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; jacobi = false),
                  Float32(cfg.Δt_split), 1.0f5)
    # A handful of real steps to get past the assembly and the zero initial guess -- not `prewarm!`:
    # this probe counts iterations rather than timing them, so there is no clock to warm, and the
    # thousands of steps a wall-clock warmup would take on a small mesh drift the Float32 time far
    # enough apart that the splitting integrator's parent/child synchronization gives up.
    for _ = 1:5
        step!(plain)
    end
    @printf("  unpreconditioned %.1f\n", mean_cg_iters!(plain, NSTEPS))
    plain = nothing
    GC.gc()

    arms = Arm[]
    # The host arms certify themselves (finite here, in band on the coarse mesh); each device arm is
    # then checked against its own host counterpart's validated state at the same step size.
    ahe, φ_host_emrkc = run_arm("host emRKC", "emRKC", form, u32, cpu_emrkc, Float32(cfg.Δt_emrkc),
                                φₘ32, nothing, false, false; steps = LV_STEPS)
    push!(arms, ahe)
    ahs, φ_host_split = run_arm("host splitting", "splitting", form, u32, cpu_split, Float32(cfg.Δt_split),
                                φₘ32, nothing, false, true; steps = LV_STEPS)
    push!(arms, ahs)
    @printf("\nhost peak RSS %.2f GiB of the 8 GiB cap (%.0f%% headroom)\n",
            host_rss_gib(), 100(1 - host_rss_gib() / 8))
    for (label, method, alg, Δt, φ_ref, has_solve) in (
        ("device emRKC", "emRKC", gpu_emrkc, Float32(cfg.Δt_emrkc), φ_host_emrkc, false),
        ("device splitting", "splitting", gpu_split, Float32(cfg.Δt_split), φ_host_split, true),
    )
        arm, _ = run_arm(label, method, form, ugpu, alg, Δt, φₘ32, φ_ref, true, has_solve; steps = LV_STEPS)
        push!(arms, arm)
        @printf("%-18s device memory in use %.2f GiB of %.2f GiB (%.0f%% headroom)\n",
                label, gpu_used_gib(), CUDA.total_memory() / 1024^3,
                100(1 - gpu_used_gib() / (CUDA.total_memory() / 1024^3)))
    end

    report(arms)
    println("\nhost-vs-device agreement is over ", LV_STEPS,
            " steps in Float32 on both sides. emRKC's step size is certified by the coarse sweep; ",
            "splitting's is a Richardson\nestimate, because its own reference does not converge at ",
            "DTREF -- see the LV-PCG2019 header block.")
    return arms
end

####################################
## Machine discipline
####################################

"""
The effective `memory.max` (bytes) of this process's own cgroup v2 leaf, read by walking
`/proc/self/cgroup`'s `0::<path>` up through `/sys/fs/cgroup<path>` until a `memory.max` file is
found. `nothing` for "max" (unset) or if no such file is found at all.
"""
function _cgroup_memory_limit()
    lines = try
        readlines("/proc/self/cgroup")
    catch
        return nothing
    end
    idx = findfirst(l -> startswith(l, "0::"), lines)
    idx === nothing && return nothing
    dir = "/sys/fs/cgroup" * split(lines[idx], "0::")[2]
    while true
        f = joinpath(dir, "memory.max")
        if isfile(f)
            v = strip(read(f, String))
            return v == "max" ? nothing : parse(Int, v)
        end
        parent = dirname(dir)
        parent == dir && return nothing
        dir = parent
    end
end

"""
Refuses to run outside a memory-capped cgroup: an uncapped run's GC sizes its heap against the whole
machine rather than the 8G this benchmark is meant to run in, which has frozen this box before.
`BENCHMARK_UNCAPPED=1` overrides for a deliberate uncapped run.
"""
function _assert_memory_capped()
    get(ENV, "BENCHMARK_UNCAPPED", "0") == "1" && return nothing
    limit = _cgroup_memory_limit()
    capped = limit !== nothing && limit ≤ 12 * 1024^3
    capped || error(
        "No memory-capped cgroup detected (effective memory.max = $(limit === nothing ? "unset" : limit) " *
        "bytes). An uncapped run's GC sizes itself against the whole machine. Run:\n" *
        "  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 " *
        "julia -t2 --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl\n" *
        "or set BENCHMARK_UNCAPPED=1 to run uncapped deliberately.",
    )
    Base.JLOptions().heap_size_hint == 0 &&
        println("WARNING: no --heap-size-hint given -- GC growth is unbounded even inside the cgroup.")
    return nothing
end

####################################

function run_model(cfg::ModelConfig)
    println("\n", "#"^118)
    println("# ", cfg.name, "  (", N, " x ", N, ", L = ", cfg.L, " mm, device Float32)")
    println("#"^118)

    form64 = ep01_form(Float64, cfg)
    u64    = ep01_u0(form64, Float64, cfg)
    φₘ64   = solution_variable(form64, :φₘ)

    println(cfg.name, ": ", Thunderbolt.solution_size(form64), " states, t ∈ [0, ", TEND, "] ms")
    @printf("reference Δt = %.4g ms, accuracy band = %.3g\n", DTREF, BAND)

    cpu64_emrkc = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    # The reference's conjugate gradient runs far tighter than the tutorial's. At the tutorial's
    # tolerances the solve's own error accumulates with the step count, so the backward Euler error
    # curve turns around below Δt ≈ 0.2 ms and a reference stepped there is *less* accurate than the
    # step sizes it is meant to certify. That floor belongs in the arms, which are configured the way
    # a user would configure them; it does not belong in the reference.
    cpu64_split = splitting(
        Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}; atol = 1.0e-12, rtol = 1.0e-10,
    )
    φ_ref_emrkc, err_emrkc = reference(form64, u64, cpu64_emrkc, φₘ64)
    φ_ref_split, err_split = reference(form64, u64, cpu64_split, φₘ64)
    @printf("  emRKC reference error ≈ %.3g ; splitting reference error ≈ %.3g\n", err_emrkc, err_split)
    @printf("  the two references differ by %.4g -- the mass lumping, not a step size\n",
            relerr(φ_ref_emrkc, φ_ref_split))

    println("\nstep size selection (Float32, host):")
    form32 = ep01_form(Float32, cfg)
    u32    = ep01_u0(form32, Float32, cfg)
    φₘ32   = solution_variable(form32, :φₘ)
    cpu32_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu32_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    Δt_emrkc, _ = select_dt(form32, u32, cpu32_emrkc, φₘ32, φ_ref_emrkc, "emRKC")
    Δt_split, _ = select_dt(form32, u32, cpu32_split, φₘ32, φ_ref_split, "splitting")

    # Both models' device arms are Float32 and genuinely so: `ParametrizedFHNModel` already wrapped its
    # literals in `T`, and `src/modeling/cells/pcg2019.jl` now does too.
    DT = Float32
    form_dev, u_dev, φₘ_dev = form32, u32, φₘ32
    gpu_emrkc = emrkc(CuVector{DT}, CuCSR)
    gpu_split = splitting(CuVector{DT}, CuCSR)
    ugpu = CuVector(u_dev)

    # What the preconditioner is worth, measured once rather than asserted: the same splitting arm at
    # the same step size with `precs` left at its default identity.
    println("\nCG iterations per step, host, Δt = ", Δt_split, " ms:")
    plain = build(form32, u32, splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; jacobi = false),
                  Float32(Δt_split), 1.0f5)
    # A handful of real steps to get past the assembly and the zero initial guess -- not `prewarm!`:
    # this probe counts iterations rather than timing them, so there is no clock to warm, and the
    # thousands of steps a wall-clock warmup would take on a small mesh drift the Float32 time far
    # enough apart that the splitting integrator's parent/child synchronization gives up.
    for _ = 1:5
        step!(plain)
    end
    @printf("  unpreconditioned %.1f\n", mean_cg_iters!(plain, NSTEPS))
    plain = nothing
    GC.gc()

    arms = Arm[]
    for (label, method, form, u0, alg, Δt, φₘ, φ_ref, dev, has_solve) in (
        ("host emRKC", "emRKC", form32, u32, cpu32_emrkc, Float32(Δt_emrkc), φₘ32, φ_ref_emrkc, false, false),
        ("host splitting", "splitting", form32, u32, cpu32_split, Float32(Δt_split), φₘ32, φ_ref_split, false, true),
        ("device emRKC", "emRKC", form_dev, ugpu, gpu_emrkc, DT(Δt_emrkc), φₘ_dev, φ_ref_emrkc, true, false),
        ("device splitting", "splitting", form_dev, ugpu, gpu_split, DT(Δt_split), φₘ_dev, φ_ref_split, true, true),
    )
        arm, _ = run_arm(label, method, form, u0, alg, Δt, φₘ, φ_ref, dev, has_solve)
        push!(arms, arm)
    end

    report(arms)
    return arms
end

# The element counts are fitted to the 8 GiB host cgroup and the 8 GiB card, not chosen for roundness
# (see the LV-PCG2019 header block); `EMRKC_LV_BASE`/`EMRKC_LV_COARSE_BASE` are what that fitting was
# done with, and what a smaller card would have to turn down.
_lv_base(key, default) = Tuple(parse.(Int, split(get(ENV, key, default), ",")))
# The step sizes are the LV's own, measured on the coarse mesh -- NOT the sheet's, which do not
# transfer (see the LV-PCG2019 block).
AVAILABLE_MODELS["LV"] = LVConfig(
    "LV-PCG2019",
    _lv_base("EMRKC_LV_BASE", "191,20,33"),
    _lv_base("EMRKC_LV_COARSE_BASE", "96,10,17"),
    0.05, 0.0068,
)

function main()
    _assert_memory_capped()
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")
    println("models: ", join(MODEL_CONFIGS, ", "),
            "  (host: 2 threads capped, ThreadedSparseMatrixCSR SpMV threaded, pointwise sweeps single-threaded)")
    for name in MODEL_CONFIGS
        try
            run_model(AVAILABLE_MODELS[name])
        catch err
            println("\n", name, " FAILED: ", sprint(showerror, err))
        end
        GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

main()
