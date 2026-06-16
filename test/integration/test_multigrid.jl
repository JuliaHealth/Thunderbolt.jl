
module TestMultigridHowto
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate-howto/multigrid.jl"))
        end
    end
end
