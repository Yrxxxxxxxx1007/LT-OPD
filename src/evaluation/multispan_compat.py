"""V8 multi-image CDPruner compatibility: prune each image span independently."""
from __future__ import annotations
import contextlib, dis, hashlib, inspect, pathlib, sys, threading, types
from collections.abc import Iterator, Mapping
from typing import Any

EXPECTED_COMPRESSOR_SOURCE_SHA256 = (
    "22a51aadcee7f08222464db0acc9250faf901444ff7e16cc8c304da7b85503a8"
)


EXPECTED_QWEN_CALLSITE_SOURCE_SHA256 = (
    "2c74d0249ed115460c9e0057319e44e71da05f303be6fb69e6d396f5c673e15c"
)


EXPECTED_COMPRESS_FUNCTION_SOURCE_SHA256 = (
    "46edba8201ff50c2d1cca34b2eb1f77d2a0aa1c6d1b5b15223c8454c281353d9"
)


EXPECTED_COMPRESS_FUNCTION_CODE_SHA256 = (
    "73835fb389cf85c247a32d70345e96ddfaa165e63ae533184ba8b6dc6f6ffebb"
)


EXPECTED_ADAPTATION_SCOPE = (
    "qwen35_single_image_post_native_merger_pre_llm_"
    "matrix_free_conditional_dpp_exact_budget"
)


EXPECTED_ALGORITHM = "qwen35_cdpruner_v1"


EXPECTED_METHOD = "cdpruner"


EXPECTED_ROUTE_KEY = "dart_merge_routes"


EXPECTED_RETENTION_BPS = 500


EXPECTED_MINIMUM_TOKENS = 32


_PATCH_LOCK = threading.RLock()


class _NonMatchingCDPrunerGuard:
    """A type that deliberately does not match the real compressor instance."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_parts(runtime: Any) -> tuple[Any, Any, Any, Any]:
    protocol = getattr(runtime, "protocol_spec", None)
    if not isinstance(protocol, Mapping) or protocol.get("algorithm") != EXPECTED_ALGORITHM:
        raise RuntimeError("multi-span overlay requires the authenticated V8 CDPruner runtime")
    if getattr(runtime, "route_key", None) != EXPECTED_ROUTE_KEY:
        raise RuntimeError("multi-span overlay runtime route key drift")
    state = getattr(runtime, "evaluation_curriculum_state", None)
    if (
        not isinstance(state, Mapping)
        or int(state.get("retention_bps", -1)) != EXPECTED_RETENTION_BPS
        or int(state.get("minimum_tokens_per_image", -1)) != EXPECTED_MINIMUM_TOKENS
    ):
        raise RuntimeError("multi-span overlay requires the bound final 5%-minimum-32 state")
    contract = getattr(runtime, "contract", None)
    compressor_config = contract.get("vision_token_compressor") if isinstance(contract, Mapping) else None
    if (
        not isinstance(compressor_config, Mapping)
        or compressor_config.get("adaptation_scope") != EXPECTED_ADAPTATION_SCOPE
        or compressor_config.get("algorithm") != EXPECTED_ALGORITHM
        or compressor_config.get("method") != EXPECTED_METHOD
        or int(compressor_config.get("minimum_tokens_per_image", -1))
        != EXPECTED_MINIMUM_TOKENS
    ):
        raise RuntimeError("multi-span overlay export compressor contract drift")

    model = getattr(runtime, "model", None)
    compressor = getattr(model, "vision_token_compressor", None)
    inner_model = getattr(model, "model", None)
    rollout = getattr(runtime, "rollout", None)
    if (
        getattr(inner_model, "vision_token_compressor", None) is not compressor
        or getattr(rollout, "module", None) is not model
    ):
        raise RuntimeError("multi-span overlay model/rollout compressor ownership drift")
    compressor_module = sys.modules.get("verl.models.transformers.vision_token_compressor")
    qwen_module = sys.modules.get("verl.models.transformers.qwen3_5")
    if compressor_module is None or qwen_module is None:
        raise RuntimeError("multi-span overlay cannot find the authenticated embedded modules")
    compressor_path = pathlib.Path(compressor_module.__file__).resolve(strict=True)
    qwen_path = pathlib.Path(qwen_module.__file__).resolve(strict=True)
    if _sha256_file(compressor_path) != EXPECTED_COMPRESSOR_SOURCE_SHA256:
        raise RuntimeError("multi-span overlay compressor source SHA-256 drift")
    if _sha256_file(qwen_path) != EXPECTED_QWEN_CALLSITE_SOURCE_SHA256:
        raise RuntimeError("multi-span overlay Qwen callsite source SHA-256 drift")

    compressor_class = getattr(compressor_module, "VisionCDPrunerCompressor", None)
    original = getattr(compressor_module, "compress_image_embeds", None)
    callsite = getattr(qwen_module, "_get_input_embeds", None)
    if not isinstance(compressor_class, type) or not isinstance(compressor, compressor_class):
        raise RuntimeError("multi-span overlay loaded compressor type drift")
    if not callable(original) or getattr(qwen_module, "compress_image_embeds", None) is not original:
        raise RuntimeError("multi-span overlay Qwen compressor callsite is not pristine")
    if not callable(callsite) or getattr(callsite, "__globals__", None) is not qwen_module.__dict__:
        raise RuntimeError("multi-span overlay Qwen callsite globals drift")
    enabled = getattr(compressor_module, "visual_token_compressor_enabled", None)
    if not callable(enabled) or enabled(model) is not True:
        raise RuntimeError("multi-span overlay authenticated compressor is not enabled")

    instructions = list(dis.get_instructions(original))
    guard_loads = [
        item
        for item in instructions
        if item.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and item.argval == "VisionCDPrunerCompressor"
    ]
    if len(guard_loads) != 1:
        raise RuntimeError("multi-span overlay expected exactly one CDPruner class guard")
    if (
        hashlib.sha256(inspect.getsource(original).encode("utf-8")).hexdigest()
        != EXPECTED_COMPRESS_FUNCTION_SOURCE_SHA256
    ):
        raise RuntimeError("multi-span overlay compressor function source SHA-256 drift")
    if (
        hashlib.sha256(original.__code__.co_code).hexdigest()
        != EXPECTED_COMPRESS_FUNCTION_CODE_SHA256
    ):
        raise RuntimeError("multi-span overlay compressor function code SHA-256 drift")
    if original.__globals__.get("VisionCDPrunerCompressor") is not compressor_class:
        raise RuntimeError("multi-span overlay compressor function globals drift")
    return compressor_module, qwen_module, original, compressor


def _derived_multispan_function(original: Any) -> Any:
    copied_globals = dict(original.__globals__)
    copied_globals["VisionCDPrunerCompressor"] = _NonMatchingCDPrunerGuard
    derived = types.FunctionType(
        original.__code__,
        copied_globals,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    derived.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    derived.__annotations__ = dict(getattr(original, "__annotations__", {}))
    derived.__dict__.update(getattr(original, "__dict__", {}))
    derived.__qualname__ = original.__qualname__
    derived.__module__ = original.__module__
    derived.__doc__ = original.__doc__
    if derived.__code__ is not original.__code__:
        raise RuntimeError("multi-span overlay failed to preserve the exact function code object")
    if set(derived.__globals__) != set(original.__globals__):
        raise RuntimeError("multi-span overlay changed the function-global inventory")
    changed_globals = {
        key
        for key in original.__globals__
        if derived.__globals__[key] is not original.__globals__[key]
    }
    if changed_globals != {"VisionCDPrunerCompressor"}:
        raise RuntimeError(
            f"multi-span overlay changed unexpected function globals: {sorted(changed_globals)}"
        )
    if derived.__globals__["visual_token_compressor_enabled"] is not original.__globals__[
        "visual_token_compressor_enabled"
    ]:
        raise RuntimeError("multi-span overlay changed the compressor enabled predicate")
    return derived


@contextlib.contextmanager
def _multispan_callsite(runtime: Any) -> Iterator[None]:
    """Install the one-call overlay and restore the pristine callsite."""

    with _PATCH_LOCK:
        _compressor_module, qwen_module, original, _compressor = _runtime_parts(runtime)
        derived = _derived_multispan_function(original)
        qwen_module.compress_image_embeds = derived
        callsite_drift = False
        try:
            yield
        finally:
            callsite_drift = qwen_module.compress_image_embeds is not derived
            qwen_module.compress_image_embeds = original
            if callsite_drift and sys.exc_info()[0] is None:
                raise RuntimeError("multi-span overlay callsite changed while generation was active")
            if qwen_module.compress_image_embeds is not original:
                raise RuntimeError("multi-span overlay failed to restore the pristine callsite")


