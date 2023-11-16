<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logo-horizontal.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-horizontal-dark.svg">
  <img alt="Thunderbolt.jl logo." src="docs/src/assets/logo-horizontal.svg">
</picture>

A modular shared-memory high-performance framework for multiscale cardiac multiphysics.

> [!WARNING]
> This package is under heavy development. Expect regular breaking changes
> for now. If you are interested in joining development, then either comment
> an issue or reach out via julialang.zulipchat.com, via mail or via 
> julialang.slack.com. Alternatively open a discussion if you have something 
> specific in mind.

## Questions

If you have questions about Thunderbolt it is suggested to use the #Thunderbolt.jl stream on Zulip.

If you encounter what you think is a bug please report it. A CONTRIBUTING.md will be provided soon with more information.

### Installation

To use Thunderbolt you first need to install Julia, see <https://julialang.org/> for details.
Installing Thunderbolt can then be done from the Pkg REPL; press `]` at the `julia>` promp to
enter `pkg>` mode:

```
pkg> add Ferrite#master, github.com/termi-official/Thunderbolt.jl#master
```

!!! note
    The package is under development, which is why you will need the currently (unreleased)
    1.0 version of Ferrite.jl and the (unregistered) Thunderbolt.jl package.

This will install Thunderbolt and all necessary dependencies. Press backspace to get back to the
`julia>` prompt. (See the [documentation for Pkg](https://pkgdocs.julialang.org/), Julia's
package manager, for more help regarding package installation and project management.)

Finally, to load Thunderbolt, use

```julia
using Thunderbolt
```

You are now all set to start using Thunderbolt!

## Accessing the documentation

Currently the documentation can only be built and viewed locally. Note that the
documentation is incomplete at this point.

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

## Community Standards

Please keep in mind that we are part of the Julia community and adhere to the Julia Community Standards.
