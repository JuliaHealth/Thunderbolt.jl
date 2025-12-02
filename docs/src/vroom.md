# Performance Tips

Many algorithms in Thunderbolt run in shared memory parallel by default if you launch julia with threads, e.g. via
```
julia --thread=<num_physical_cores>
```

In our experience exceeding the number of physical cores breaks performance for most simulations.
Depending on your cache size simulations can even run faster if not all physical cores are utilized.

We also recommend to pin to cores for threaded simulations.
```julia
using ThreadPinning
pinthreads(:cores)
```
