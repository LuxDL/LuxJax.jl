# LuxJax

LuxJax allows you to use Neural Networks written in Jax with the Lux API, allowing seamless
integration with the rest of the SciML ecosystem.

Lux.jl is great and is quite fast and useful if you are implementing custom operations.
However, there are quite a few standard workloads where XLA can highly optimize the training
and inference. This package bridges that gap, and allows you to use the fast Jax Neural
Networks with the SciMLverse!

## Installation

The installation is currently a bit manual. First install this package.

```julia
import Pkg
Pkg.add("https://github.com/LuxDL/LuxJax.jl")
```

Then install the Jax dependencies.

```julia
using LuxJax
LuxJax.install("<setup>")
```

`install` will install the Jax dependencies based on the `setup` provided!

## Usage Example

```julia
using LuxJax
```

## Tips

* When mixing jax and julia it's recommended to disable jax's preallocation with setting the
  environment variable `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Roadmap

- [ ] Automatic Differentiation
    - [ ] Capture Chain Rules
        - [ ] Reverse Mode
        - [ ] Forward Mode (Very Low Priority)
    - [ ] Capture ForwardDiff Duals for Forward Mode
- [ ] Automatically Map Lux Models to Flax Models similar to the Flux to Lux conversion
- [ ] Handle Component Arrays
- [ ] Demonstrate Training of Neural ODEs using Jax and SciMLSensitivity.jl

## Acknowledgements

This packages is a more opinionated take on
[PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)
