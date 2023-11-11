using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        Thunderbolt;
        ambiguities=false, # Tested below for now
        # unbound_args=true,
        # undefined_exports=true,
        # project_extras=true,
        stale_deps=(ignore=[:FerriteGmsh],), # We use this one in the examples for now 
        deps_compat=false,
        # piracy=false,
    )
    Aqua.test_ambiguities(Thunderbolt) # Must be separate for now, see https://github.com/JuliaTesting/Aqua.jl/issues/77
end
