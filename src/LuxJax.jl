module LuxJax

using CondaPkg, PythonCall # Interact with Jax
using DLPack               # Transfer Data between Julia and Jax
using ChainRulesCore, Functors, Random
using LuxCore
import ConcreteStructs: @concrete

# Setup Jax
const jax = PythonCall.pynew()
const dlpack = PythonCall.pynew()
const jnp = PythonCall.pynew()
const numpy = PythonCall.pynew()
const flax = PythonCall.pynew()
const linen = PythonCall.pynew()

const is_jax_setup = Ref{Bool}(false)

const VALID_JAX_SETUPS = ("cpu", "cuda12_pip", "cuda11_pip", "cuda12_local", "cuda11_local",
    "tpu")

function __load_jax_dependencies(; force::Bool = true)
    try
        CondaPkg.withenv() do
            PythonCall.pycopy!(jax, pyimport("jax"))
            PythonCall.pycopy!(dlpack, pyimport("jax.dlpack"))
            PythonCall.pycopy!(jnp, pyimport("jax.numpy"))
            PythonCall.pycopy!(flax, pyimport("flax"))
            PythonCall.pycopy!(linen, pyimport("flax.linen"))
        end

        is_jax_setup[] = true
    catch err
        is_jax_setup[] = false

        if force
            rethrow(err)
        else
            @warn "Jax is not installed. Please install Jax first using `LuxJax.install(<setup>)`!"
            @debug err
        end
    end
end

function __init__()
    CondaPkg.withenv() do
        PythonCall.pycopy!(numpy, pyimport("numpy"))
    end
    __load_jax_dependencies(; force = false)
end

"""
    install(setup::String = "cpu")

Installs Jax into the correct environment. The `setup` argument can be one of the following:

  - `"cpu"`: Installs Jax with CPU support.
  - `"cuda12_pip"`: Installs Jax with CUDA 12 support using pip.
  - `"cuda11_pip"`: Installs Jax with CUDA 11 support using pip.
  - `"cuda12_local"`: Installs Jax with CUDA 12 support using a local CUDA installation.
  - `"cuda11_local"`: Installs Jax with CUDA 11 support using a local CUDA installation.
  - `"tpu"`: Installs Jax with TPU support.
"""
function install(setup::String = "cpu")
    @assert setup ∈ VALID_JAX_SETUPS "Invalid setup: $(setup)! Select one of $(VALID_JAX_SETUPS)!"
    CondaPkg.withenv() do
        python = CondaPkg.which("python")
        run(`$(python) --version`)
        run(`$(python) -m pip install --upgrade pip`)
        if occursin("cpu", setup)
            run(`$(python) -m pip install --upgrade jax\[cpu\]\>=0.4`)
        elseif occursin("cuda", setup)
            run(`$(python) -m pip install --upgrade jax\[$(setup)\]\>=0.4 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`)
        else # TPU
            run(`$(python) -m pip install --upgrade jax\[$(setup)\]\>=0.4 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`)
        end
    end

    __load_jax_dependencies()

    return
end

# FIXME: Wrap DLPack functions to allow them to work for tuple, namedtuple and functors

# Lux Interop
__from_flax_params(ps) = (; params = Tuple(map(p -> DLPack.wrap(p, dlpack.to_dlpack), ps)))

function __to_flax_params(ps, tree)
    return jax.tree_util.tree_unflatten(tree,
        pylist(map(p -> DLPack.share(p, dlpack.from_dlpack), ps.params)))
end

@concrete struct LuxFlaxWrapper <: LuxCore.AbstractExplicitLayer
    flaxmodel
    input_shape
    tree_structure
end

function LuxFlaxWrapper(flaxmodel, input_shape)
    return LuxFlaxWrapper(flaxmodel, input_shape, PythonCall.pynew())
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::LuxFlaxWrapper)
    seed = rand(rng, UInt32)
    params = layer.flaxmodel.init(jax.random.PRNGKey(seed),
        jnp.ones((1, reverse(layer.input_shape)...)))
    ps_flat, tree_structure = jax.tree_util.tree_flatten(params)
    PythonCall.pycopy!(layer.tree_structure, tree_structure)
    return __from_flax_params(ps_flat)
end

(l::LuxFlaxWrapper)(x, ps, st) = LuxCore.apply(l, x, ps, st)

function LuxCore.apply(l::LuxFlaxWrapper, x, ps, st::NamedTuple)
    x_jax = DLPack.share(x, dlpack.from_dlpack)
    ps_jax = __to_flax_params(ps, l.tree_structure)
    y = l.flaxmodel.apply(ps_jax, x_jax)
    return DLPack.wrap(y, dlpack.to_dlpack), st
end

function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::LuxFlaxWrapper, x, ps,
    st::NamedTuple)
    projectₓ = ProjectTo(x)
    projectₚ = ProjectTo(ps)
    x_jax = DLPack.share(x, dlpack.from_dlpack)
    ps_jax = __to_flax_params(ps, l.tree_structure)
    y, jax_vjpfun = jax.vjp(l.flaxmodel.apply, ps_jax, x_jax)
    function ∇flax_apply(Δ)
        # FIXME: Fix dispatches so that we dont have to collect
        ∂y = DLPack.share(collect(first(unthunk(Δ))), dlpack.from_dlpack)
        (∂ps_jax, ∂x_jax) = jax_vjpfun(∂y)
        ∂x = projectₓ(DLPack.wrap(∂x_jax, dlpack.to_dlpack))
        ∂ps = projectₚ(__from_flax_params(jax.tree_util.tree_flatten(∂ps_jax)[0]))
        return (NoTangent(), NoTangent(), ∂x, ∂ps, NoTangent())
    end
    return (DLPack.wrap(y, dlpack.to_dlpack), st), ∇flax_apply
end

# exports
export jax, jnp, flax, linen
export LuxFlaxWrapper

end
