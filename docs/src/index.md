```@meta
DocTestSetup = :(using Thunderbolt)
```
# Thunderbolt.jl

*A high performance cardiac multiphysics system written in Julia*

Welcome to the documentation for Thunderbolt. The main goal of this project
is to provide a single framework where we can develop new models and high 
performance parallel solvers.

!!! warning
    This package is under heavy development. Expect regular breaking changes
    for now. If you are interested in joining development, then either comment
    an issue or reach out via julialang.zulipchat.com, via mail or via 
    julialang.slack.com. Alternatively open a discussion if you have something 
    specific in mind.

!!! note
    If you are interested in using this package, then I am also happy to
    to get some constructive feedback, especially if things don't work out
    in the current design. This can be done via julialang.slack.com,
    julialang.zulipchat.com or via mail.

!!! note
    If you use this package in an academic context, then I would be happy if
    you could cite it. Please also cite additionally the corresponding sources
    for models, numerical methods and utilities used in your code via this package.

## How the documentation is organized

This high level view of the documentation structure will help you find what you are looking
for. The document is organized as follows[^1]:

 - [**Tutorials**](tutorials/index.md) are thoroughly documented examples which guides you
   through the process of building and solving cardiac models in Thunderbolt.
 - [**Topic guides**](topics/index.md) contains more in-depth explanations and discussions
   about multiphysics modeling concepts and their numerical treatment, and specifically how 
   these are realized in Thunderbolt.
 - [**API Reference**](api-reference/index.md) contains the technical API reference of functions and
   methods (e.g. the documentation strings).
 - [**How-to guides**](howto/index.md) will guide you through the steps involved in
   addressing common tasks and use-cases. These usually build on top of the tutorials and
   thus assume basic knowledge of how Thunderbolt works.

[^1]: The organization of the document follows the [Diátaxis Framework](https://diataxis.fr).

In addition there is the [**Developer documentation**](devdocs/index.md), for documentation of
Ferrite internal code.


## Getting started

If you are new to the Thunderbolt project, it is suggested to start with the tutorials
section before tackling more complex problems.

!!! todo
    We should provide a template for computational studies, possibly via DrWatson.jl ([DrWatsonSim.jl](https://github.com/sebastianpech/DrWatsonSim.jl)?), to guide users towards best practices for computational studies.

### Getting help

If you have questions about Thunderbolt it is suggested to use the `#Thunderbolt.jl` stream on
[Zulip](https://julialang.zulipchat.com/). Zulip is preferred over Slack, because the discussions
are available over longer time periods.

### Installation

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

## Contributing to Thunderbolt

Thunderbolt is under very active development. If you find a bug, then please open an [issue on GitHub](https://github.com/JuliaHealth/Thunderbolt.jl/issues) with a reproducer.
If you are interested in joining development, then either comment an issue or reach out via [Zulip](https://julialang.zulipchat.com), via mail or via 
[Slack](https://julialang.slack.com). Alternatively open a [discussion](https://github.com/JuliaHealth/Thunderbolt.jl/discussions) if you have something 
specific in mind - please just check for open discussion before opening a new one.

A detailed contributor guide is coming soon...
