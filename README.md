<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logo-horizontal.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-horizontal-dark.svg">
  <img alt="Thunderbolt.jl logo." src="docs/src/assets/logo-horizontal.svg">
</picture>

A modular shared-memory high-performance framework for multiscale cardiac multiphysics.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaHealth.github.io/Thunderbolt.jl/dev)

> [!WARNING]
> This package is under heavy development. Expect regular breaking changes
> for now. If you are interested in joining development, then either comment
> an issue you are interested or directly reach out to me via julialang.zulipchat.com, mail or
> julialang.slack.com. Alternatively open a [GitHub discussion](https://github.com/JuliaHealth/Thunderbolt.jl/discussions) if you have something
> specific in mind.

## Questions

If you have questions about Thunderbolt it is suggested to use the [#Thunderbolt.jl stream on Zulip](https://julialang.zulipchat.com/#narrow/stream/413315-Thunderbolt.2Ejl).

If you encounter what you think is a bug please report it. A CONTRIBUTING.md will be provided soon with more information.

## Installation

To use Thunderbolt you first need to install Julia, see <https://julialang.org/> for details.
Installing Thunderbolt can then be done from the Pkg REPL; press `]` at the `julia>` promp to
enter `pkg>` mode:

```
pkg> add https://github.com/JuliaHealth/Thunderbolt.jl#main
```

This will install Thunderbolt and all necessary dependencies. Press backspace to get back to the
`julia>` prompt. (See the [documentation for Pkg](https://pkgdocs.julialang.org/), Julia's
package manager, for more help regarding package installation and project management.)

Finally, to load Thunderbolt, use

```julia
using Thunderbolt
```

You are now all set to start using Thunderbolt!

## Accessing the Documentation Offline

To build the docs you first need to install Julia, see <https://julialang.org/> for details.
Then you need to install the packages. To accomplish this you first start a julia shell in the
docs fonder and open the Pkg REPL; i.e. press `]` at the `julia>` promp to
enter `pkg>` mode. Then you [activate and instantiate](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project) via
```
(@v1.9) pkg> activate .
(docs) pkg> instantiate
```
This has to be done only once. Now you can use the provided liveserver to build and open the docs via
```
$ julia --project=. liveserver.jl
```
Now you can view the documentation at https://localhost:8000 . All further information will be provided there.

## How to Cite

If you use this package, please cite our repository.

> Dennis Ogiermann. **Thunderbolt.jl v0.0.1**. [2025]. GitHub repository. https://github.com/JuliaHealth/Thunderbolt.jl

## Community Standards

Please keep in mind that we are part of the Julia community and adhere to the [Julia Community Standards](https://julialang.org/community/standards/).

## Acknowledgement

This framework is primarily developed at the [chair of continuum mechanics at Ruhr University Bochum under Prof. Dr.-Ing. Daniel Balzani](https://www.lkm.ruhr-uni-bochum.de/).
