"""LT-OPD CDPruner inference using the training rollout implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import sys
from types import MethodType, SimpleNamespace


EXPORT_SCHEMA = "vision_opd_ai4s_v6_holitom_dpc_export_v1"
ALGORITHM = "qwen35_holitom_dpc_spatial_merge_v1"
METHOD = "holitom_inspired_dpc_diversity_merge"
ROUTE_SCHEMA = "vision_opd_holitom_dpc_merge_route_v1"
ROUTE_KEY = "dpc_merge_routes"
NO_IMAGE_POLICY = "same_prompt_image_placeholder_without_visual_replacement_v1"
REFERENCE_TARGET_POLICY = "assistant_content_plus_im_end_lf_v1"
V8_EXPORT_SCHEMA = "vision_opd_ai4s_v8_cdpruner_export_v1"
V8_STATIC_SCHEMA = "vision_opd_ai4s_v8_fullimage_curriculum_static_contract_v1"
V8_REFERENCE_TARGET_POLICY = "assistant_content_plus_answer_close_no_eos_v1"
V8_RESPONSE_PROTOCOL_SCHEMA = "vision_opd_ai4s_v8_virtual_open_answer_transport_v1"
V8_ROLLOUT_UID_SCHEMA = "vision_opd_ai4s_v8_export_rollout_uid_v1"
V8_FINAL_DEPLOYMENT_ROLE = "final_deployment"
V8_DEFERRED_DIAGNOSTIC_ROLE = "deferred_diagnostic_panel_only"
V8_DIAGNOSTIC_AUTHORIZATION_SCHEMA = (
    "vision_opd_ai4s_v8_deferred_diagnostic_export_authorization_v1"
)
EXPORT_PROTOCOLS_BY_SCHEMA = {
    EXPORT_SCHEMA: {
        "algorithm": ALGORITHM,
        "method": METHOD,
        "route_schema": ROUTE_SCHEMA,
        "route_key": ROUTE_KEY,
        "reference_target_policy": REFERENCE_TARGET_POLICY,
        "virtual_open": False,
        "transport_prefix": "",
    },
    V8_EXPORT_SCHEMA: {
        "algorithm": "qwen35_cdpruner_v1",
        "method": "cdpruner",
        "route_schema": "vision_opd_cdpruner_curriculum_route_v2",
        "route_key": "dart_merge_routes",
        "reference_target_policy": V8_REFERENCE_TARGET_POLICY,
        "virtual_open": True,
        "transport_prefix": "<answer>",
    },
}
FORBIDDEN_RUNTIME_ENV = (
    "RAY_ADDRESS",
    "RAY_JOB_CONFIG_JSON",
    "PYTHONPATH",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "FLASH_ATTENTION_DETERMINISTIC",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_CUMEM_ENABLE",
    "VERL_EXP3_BUNDLED_RUNTIME_ROOT",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()








def _contract_spec(contract):
    schema = contract.get("schema_version")
    spec = EXPORT_PROTOCOLS_BY_SCHEMA.get(schema)
    if spec is None:
        raise RuntimeError(f"unsupported DPC export schema: {schema!r}")
    if schema == V8_EXPORT_SCHEMA:
        static_binding = contract.get("static_contract")
        if (
            not isinstance(static_binding, dict)
            or set(static_binding)
            != {"schema_version", "canonical_sha256", "file_sha256"}
            or static_binding.get("schema_version") != V8_STATIC_SCHEMA
            or re.fullmatch(r"[0-9a-f]{64}", str(static_binding.get("canonical_sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(static_binding.get("file_sha256", ""))) is None
        ):
            raise RuntimeError("V8 export lacks its exact static-contract schema binding")
    return spec


def _require_v8_final_evaluation_state(compressor, expected_state):
    """Construct and authenticate the one permitted V8 inference state."""

    from verl.models.transformers.visual_token_curriculum import VisualTokenCurriculum

    curriculum = VisualTokenCurriculum.from_mapping(compressor.get("curriculum"))
    actual = curriculum.runtime_state(curriculum.total_optimizer_steps - 1)
    if not isinstance(expected_state, dict) or expected_state != actual:
        raise RuntimeError(
            "V8 export does not bind the exact final curriculum evaluation state"
        )
    return actual


def _decode_masked_response_rows(tokenizer, responses, response_mask):
    """Decode only sampled actions, preserving tokenizer whitespace bytes."""

    import torch

    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise RuntimeError("responses and response_mask must be equal rank-2 tensors")
    mask = response_mask.to(dtype=torch.long)
    if not torch.equal(mask, mask.to(torch.bool).to(torch.long)):
        raise RuntimeError("response_mask must contain only binary values")
    if bool((mask[:, 1:] > mask[:, :-1]).any().item()):
        raise RuntimeError("response_mask must be a contiguous prefix on every row")
    if bool((mask.sum(dim=-1) <= 0).any().item()):
        raise RuntimeError("every generated row must contain at least one sampled action")
    decoded = []
    for row, row_mask in zip(responses, mask, strict=True):
        active_ids = row[row_mask.to(torch.bool)].detach().cpu().tolist()
        decoded.append(
            tokenizer.decode(
                active_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return decoded


def _bind_route_identity(routes_by_sample, spec, *, evaluation_curriculum_state=None):
    """Attach/validate the release identity on every emitted route payload."""

    if not spec["virtual_open"]:
        return routes_by_sample
    if not isinstance(routes_by_sample, (list, tuple)):
        # HFRollout returns an object ndarray for batched route lists.
        try:
            routes_by_sample = list(routes_by_sample)
        except TypeError as exc:
            raise RuntimeError("V8 route transport is not batch-iterable") from exc
    expected = {
        "schema_version": spec["route_schema"],
        "algorithm": spec["algorithm"],
        "method": spec["method"],
    }
    if spec["virtual_open"]:
        if not isinstance(evaluation_curriculum_state, dict):
            raise RuntimeError("V8 route validation requires the bound final curriculum state")
        expected_route_state = {
            "retention_bps": evaluation_curriculum_state["retention_bps"],
            "curriculum_completed_steps": evaluation_curriculum_state[
                "completed_optimizer_steps"
            ],
            "curriculum_schedule_sha256": evaluation_curriculum_state["schedule_sha256"],
        }
    for sample_routes in routes_by_sample:
        if not isinstance(sample_routes, list) or not sample_routes:
            raise RuntimeError("V8 route transport contains an empty/non-list sample")
        for route in sample_routes:
            if not isinstance(route, dict):
                raise RuntimeError("V8 route payload must be a dictionary")
            drift = {
                key: {"expected": value, "actual": route.get(key)}
                for key, value in expected.items()
                if route.get(key) != value
            }
            if drift:
                raise RuntimeError(f"V8 CDPruner route identity drift: {drift}")
            state_drift = {
                key: {"expected": value, "actual": route.get(key)}
                for key, value in expected_route_state.items()
                if route.get(key) != value
            }
            if state_drift:
                raise RuntimeError(f"V8 CDPruner final-evaluation route state drift: {state_drift}")
            from verl.models.transformers.vision_token_compressor import (
                DARTMergeRoute,
                validate_cdpruner_curriculum_route,
            )

            replayable = {key: value for key, value in route.items() if key != "query_audit"}
            validate_cdpruner_curriculum_route(
                DARTMergeRoute.from_dict(replayable), evaluation_curriculum_state
            )
    return routes_by_sample


def _canonical_v8_rollout_uid(rendered_prompt, canonical_route_query, sample_index):
    """Bind one export-runtime sample to deterministic, process-independent bytes."""

    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise ValueError("V8 rollout UID requires a non-empty canonical rendered prompt")
    if not isinstance(canonical_route_query, dict):
        raise TypeError("V8 rollout UID requires a canonical route-query mapping")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("V8 rollout UID sample_index must be a non-negative integer")
    payload = {
        "schema_version": V8_ROLLOUT_UID_SCHEMA,
        "sample_index": sample_index,
        "rendered_prompt": rendered_prompt,
        "route_query": canonical_route_query,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_v8_route_query_audits(
    routes_by_sample, route_queries_batch, sample_uids_batch
):
    """Cross-bind every emitted CD route to its canonical producer query."""

    canonical_queries = _canonical_v8_route_queries(
        route_queries_batch, len(routes_by_sample)
    )
    if isinstance(sample_uids_batch, (str, bytes)):
        raise TypeError("V8 CDPruner sample UIDs must be a batch-aligned sequence")
    try:
        sample_uids = list(sample_uids_batch)
    except TypeError as exc:
        raise TypeError("V8 CDPruner sample UIDs must be a batch-aligned sequence") from exc
    if len(sample_uids) != len(routes_by_sample) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in sample_uids
    ):
        raise RuntimeError("V8 CDPruner sample UIDs are missing, malformed, or not batch aligned")
    if len(set(sample_uids)) != len(sample_uids):
        raise RuntimeError("V8 CDPruner sample UIDs must be unique within one runtime batch")
    exact_fields = {
        "schema_version",
        "query_policy",
        "source",
        "canonical_text",
        "canonical_sha256",
        "segments",
        "semantic_character_spans",
        "selected_token_count",
        "selected_token_indices",
        "selected_token_ids",
        "prefill_left_padding",
        "prefill_prompt_length",
        "sample_uid",
    }
    for sample_routes, query, sample_uid in zip(
        routes_by_sample, canonical_queries, sample_uids, strict=True
    ):
        expected = {
            "schema_version": query["schema_version"],
            "query_policy": query["policy"],
            "source": "route_query",
            "canonical_text": query["canonical_text"],
            "canonical_sha256": query["sha256"],
            "segments": query["segments"],
            "sample_uid": sample_uid,
        }
        for route in sample_routes:
            audit = route.get("query_audit")
            if not isinstance(audit, dict) or set(audit) != exact_fields:
                raise RuntimeError("V8 CDPruner route has a non-canonical query-audit inventory")
            drift = {
                key: {"expected": value, "actual": audit.get(key)}
                for key, value in expected.items()
                if audit.get(key) != value
            }
            indices = audit.get("selected_token_indices")
            token_ids = audit.get("selected_token_ids")
            spans = audit.get("semantic_character_spans")
            count = audit.get("selected_token_count")
            prompt_length = audit.get("prefill_prompt_length")
            left_padding = audit.get("prefill_left_padding")
            structurally_valid = (
                not drift
                and isinstance(indices, list)
                and isinstance(token_ids, list)
                and isinstance(spans, list)
                and len(spans) == len(query["segments"])
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                and count == len(indices) == len(token_ids)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in indices)
                and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_ids)
                and indices == sorted(set(indices))
                and isinstance(prompt_length, int)
                and not isinstance(prompt_length, bool)
                and prompt_length > 0
                and all(0 <= value < prompt_length for value in indices)
                and isinstance(left_padding, int)
                and not isinstance(left_padding, bool)
                and 0 <= left_padding < prompt_length
                and all(
                    isinstance(span, list)
                    and len(span) == 2
                    and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
                    and 0 <= span[0] < span[1]
                    for span in spans
                )
            )
            if not structurally_valid:
                raise RuntimeError(
                    f"V8 CDPruner route/query-audit binding drift: {drift}"
                )
    return routes_by_sample


def _reference_target_text(reference, spec):
    """Return the exact assistant continuation owned by the scoring target."""

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("every reference answer must be a non-empty string")
    # Keep the historical V6/V7 rejection surface byte-for-byte: those
    # releases rejected chat-boundary injection but did not special-case the
    # tokenizer pad token.  V8 owns a stricter content-only transport contract.
    if "<|im_start|>" in reference or "<|im_end|>" in reference:
        raise ValueError("reference answers may not inject Qwen chat-control tokens")
    if spec["virtual_open"]:
        folded_reference = reference.casefold()
        if (
            "<answer>" in folded_reference
            or "</answer>" in folded_reference
            or "<|endoftext|>" in reference
            or "<|" in reference
            or "|>" in reference
        ):
            raise ValueError("V8 reference answers must contain content only, not protocol tags")
        return reference + "</answer>"
    return reference + "<|im_end|>\n"


def _extend_prompt_query_mask(prompt_query_mask, target):
    """Extend a prompt-only CD query mask with an all-false target suffix."""

    import torch

    if (
        prompt_query_mask is None
        or prompt_query_mask.ndim != 2
        or prompt_query_mask.dtype != torch.bool
        or target.ndim != 2
        or prompt_query_mask.shape[0] != target.shape[0]
        or int(prompt_query_mask.sum().item()) <= 0
    ):
        raise RuntimeError("CDPruner scoring requires a non-empty boolean prompt query mask")
    result = torch.cat(
        (prompt_query_mask, torch.zeros_like(target, dtype=torch.bool)), dim=-1
    )
    if bool(result[:, -target.shape[-1] :].any().item()):
        raise RuntimeError("reference target tokens leaked into the CDPruner query mask")
    return result


def _canonical_v8_route_queries(route_queries_batch, batch_size):
    """Validate exact, batch-aligned V2 route-query transport mappings."""

    from verl.utils.route_query import (
        ROUTE_QUERY_SCHEMA_VERSION_V2,
        parse_route_query_value,
    )

    if not isinstance(route_queries_batch, list) or len(route_queries_batch) != batch_size:
        raise ValueError(
            "V8 CDPruner runtime requires batch-aligned route_queries_batch for every mode"
        )
    normalized = []
    for raw_query in route_queries_batch:
        if not isinstance(raw_query, dict):
            raise TypeError("every V8 route query must be a canonical V2 mapping")
        parsed = parse_route_query_value(raw_query)
        canonical = parsed.as_data_field()
        if parsed.schema_version != ROUTE_QUERY_SCHEMA_VERSION_V2 or raw_query != canonical:
            raise ValueError("every V8 route query must be the exact canonical V2 data form")
        normalized.append(canonical)
    return normalized




def _load_and_validate_contract(export_dir, **kwargs):
    export_dir = pathlib.Path(export_dir).resolve()
    contract_path = export_dir / "cdpruner_export_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != V8_EXPORT_SCHEMA:
        raise ValueError("Expected an LT-OPD V8 export.")
    _contract_spec(contract)
    if contract.get("artifact_role") != "final_deployment" or contract.get("checkpoint_step") != 175:
        raise ValueError("Evaluation requires the final 175-step checkpoint.")
    compressor = contract["vision_token_compressor"]
    if compressor.get("algorithm") != "qwen35_cdpruner_v1":
        raise ValueError("Expected the CDPruner runtime.")
    _require_v8_final_evaluation_state(compressor, contract.get("evaluation_curriculum_state"))
    for name in ("model.safetensors", "v6_dpc_config.json", "tokenizer.json"):
        if not (export_dir / "compressed_actor_model" / name).is_file():
            raise FileNotFoundError(name)
    return export_dir, contract


def _require_cuda_model_residency(model, requested_device_map):
    observed = set()
    for values in (model.parameters(), model.buffers()):
        for value in values:
            observed.add(str(value.device))
            if value.device.type == "cuda":
                return
    raise RuntimeError(
        "release-06 DPC uses the audited CUDA/FlashAttention runtime; "
        f"requested device_map={requested_device_map!r}, observed={sorted(observed)!r}"
    )


def _install_bundled_verl(export_dir):
    # The installed LT-OPD package supplies the same pruning and rollout code.
    import verl
    return pathlib.Path(verl.__file__).resolve().parent.parent


def _attach_dpc(model, compressor_config, *, evaluation_curriculum_state=None):
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.models.transformers.vision_token_compressor import build_vision_token_compressor

    apply_monkey_patch(
        model,
        ulysses_sp_size=1,
        use_remove_padding=False,
        use_fused_kernels=False,
        use_prefix_grouper=False,
    )
    compressor = build_vision_token_compressor(compressor_config, allow_legacy=False)
    algorithm = compressor_config.get("algorithm")
    if algorithm not in {ALGORITHM, "qwen35_cdpruner_v1"} or compressor.algorithm != algorithm:
        raise RuntimeError("bundled loader reconstructed a non-DPC compressor")
    active_curriculum_state = None
    if hasattr(compressor, "set_final_retention_for_evaluation"):
        active_curriculum_state = compressor.set_final_retention_for_evaluation()
        expected_state = _require_v8_final_evaluation_state(
            compressor_config, evaluation_curriculum_state
        )
        if active_curriculum_state != expected_state:
            raise RuntimeError("V8 compressor did not enter the bound final evaluation state")
    model.vision_token_compressor = compressor
    model.vision_token_compressor_enabled = True
    model.vision_token_compressor_algorithm = algorithm
    model.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
    model.vision_token_compressor_retention_ratio = compressor.retention_ratio
    model.vision_token_compressor_retention_bps = getattr(compressor, "active_retention_bps", 500)
    model.model.vision_token_compressor = compressor
    model.model.vision_token_compressor_enabled = True
    model.model.vision_token_compressor_algorithm = algorithm
    model.model.vision_token_compressor_minimum_tokens = compressor.minimum_tokens
    model.model.vision_token_compressor_retention_ratio = compressor.retention_ratio
    model.model.vision_token_compressor_retention_bps = getattr(compressor, "active_retention_bps", 500)
    if active_curriculum_state is not None:
        model.vision_token_compressor_evaluation_curriculum_state = dict(active_curriculum_state)
        model.model.vision_token_compressor_evaluation_curriculum_state = dict(active_curriculum_state)
    model.model.vision_token_compressor_last_routes = None
    model.config.vision_token_compressor = compressor_config


def prepare_loaded_model_for_v6_dpc_runtime(
    model, compressor_config, *, attach_dpc=True, evaluation_curriculum_state=None
):
    """Install the formal compressor and guarded functional generation API.

    This is also used by the separately hash-bound base-initialization arm;
    it never loads, copies, or mutates the source checkpoint on disk.
    """

    if attach_dpc:
        _attach_dpc(
            model,
            compressor_config,
            evaluation_curriculum_state=evaluation_curriculum_state,
        )
    model.requires_grad_(False).eval()
    original_generate = getattr(model, "_v6_dpc_transformers_generate", None)
    if original_generate is None:
        original_generate = model.generate
        model._v6_dpc_transformers_generate = original_generate
    if not callable(original_generate):
        raise RuntimeError("loaded actor has no callable Transformers generation entry point")

    def _forbid_stock_generate(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "stock model.generate is invalid for a physically compressed cache; "
            "use the V6 functional runtime"
        )

    model.generate = _forbid_stock_generate
    return model


def _load_validated_v6_dpc_model(export_dir, contract, *, torch_dtype, device_map):
    _install_bundled_verl(export_dir)
    from transformers import AutoConfig, AutoModelForImageTextToText

    model_dir = export_dir / "compressed_actor_model"
    raw_config = json.loads((model_dir / "v6_dpc_config.json").read_text(encoding="utf-8"))
    spec = _contract_spec(contract)
    if (
        raw_config.get("vision_opd_requires_v6_dpc_runtime") is not True
        or raw_config.get("v6_dpc_export_contract") != "../cdpruner_export_contract.json"
        or raw_config.get("vision_token_compressor") != contract["vision_token_compressor"]
        or (
            spec["virtual_open"]
            and raw_config.get("vision_opd_static_contract_schema") != V8_STATIC_SCHEMA
        )
    ):
        raise RuntimeError("guarded model configuration differs from the V6 DPC export contract")
    model_type = raw_config.pop("model_type")
    config = AutoConfig.for_model(model_type, **raw_config)
    if config.model_type != "qwen3_5":
        raise RuntimeError(f"release-06 export requires qwen3_5, got {config.model_type!r}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=contract["attn_implementation"],
        trust_remote_code=True,
    )
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    expected_fp32 = {
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc2.bias",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.norm.bias",
        "model.visual.merger.norm.weight",
    }
    weight_path = model_dir / "model.safetensors"
    with safe_open(str(weight_path), framework="pt", device="cpu") as checkpoint:
        checkpoint_dtypes = {
            name: str(checkpoint.get_slice(name).get_dtype()) for name in checkpoint.keys()
        }
        observed_fp32 = {
            name for name, dtype_name in checkpoint_dtypes.items() if dtype_name == "F32"
        }
        unexpected_dtypes = {
            name: dtype_name
            for name, dtype_name in checkpoint_dtypes.items()
            if dtype_name not in {"BF16", "F32"}
        }
        if observed_fp32 != expected_fp32 or unexpected_dtypes:
            raise RuntimeError(
                "bound actor mixed-dtype inventory drift: "
                f"expected_fp32={sorted(expected_fp32)}, "
                f"observed_fp32={sorted(observed_fp32)}, "
                f"unexpected_dtypes={unexpected_dtypes}"
            )
        parameters = dict(model.named_parameters())
        if not expected_fp32.issubset(parameters):
            raise RuntimeError(
                "loaded actor lacks bound FP32 visual-merger parameters: "
                f"{sorted(expected_fp32 - set(parameters))}"
            )
        for name in sorted(expected_fp32):
            parameter = parameters[name]
            value = checkpoint.get_tensor(name)
            if value.dtype != torch.float32:
                raise RuntimeError(f"bound FP32 tensor decoded with wrong dtype: {name}")
            set_module_tensor_to_device(
                model,
                name,
                parameter.device,
                value=value,
                dtype=torch.float32,
            )
    restored = dict(model.named_parameters())
    if any(restored[name].dtype != torch.float32 for name in expected_fp32):
        raise RuntimeError("failed to restore the bound FP32 visual-merger parameters")
    _require_cuda_model_residency(model, device_map)
    return prepare_loaded_model_for_v6_dpc_runtime(
        model,
        contract["vision_token_compressor"],
        attach_dpc=True,
        evaluation_curriculum_state=(
            contract.get("evaluation_curriculum_state") if spec["virtual_open"] else None
        ),
    )


def load_v6_dpc_model(export_dir, *, torch_dtype="auto", device_map="auto"):
    """Load only the hash-bound compressed actor; stock generation is disabled."""

    export_dir, contract = _load_and_validate_contract(export_dir)
    return _load_validated_v6_dpc_model(
        export_dir,
        contract,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


class V6DPCInferenceRuntime:
    """Public wrapper around the embedded DPC cached-generation implementation."""

    def __init__(self, export_dir, model, processor, runtime_contract):
        from verl.workers.config import RolloutConfig
        from verl.workers.rollout.hf_rollout import HFRollout
        from verl.utils.model import align_qwen35_chat_generation_config

        class AuditedHFRollout(HFRollout):
            def _sample_dart_next_token(self, logits, **kwargs):
                if getattr(self, "_v6_capture_level", "none") == "full":
                    detached = logits.detach().float().cpu()
                    self._v6_decode_logits.append(detached)
                    self._v6_greedy_tokens.append(detached.argmax(dim=-1))
                return super()._sample_dart_next_token(logits, **kwargs)

        if not isinstance(runtime_contract, dict):
            raise TypeError("runtime_contract must be a dictionary")
        if "vision_token_compressor" in runtime_contract:
            self.contract = runtime_contract
            compressor_config = runtime_contract["vision_token_compressor"]
            static_schema = runtime_contract.get("static_contract", {}).get("schema_version")
            if static_schema == V8_STATIC_SCHEMA:
                self.protocol_spec = EXPORT_PROTOCOLS_BY_SCHEMA[V8_EXPORT_SCHEMA]
            else:
                self.protocol_spec = EXPORT_PROTOCOLS_BY_SCHEMA[EXPORT_SCHEMA]
        else:
            # Compatibility for the historical base-runtime constructor.  It
            # cannot activate V8 because no response protocol is available.
            self.contract = {"vision_token_compressor": runtime_contract}
            compressor_config = runtime_contract
            self.protocol_spec = EXPORT_PROTOCOLS_BY_SCHEMA[EXPORT_SCHEMA]
        self.route_key = self.protocol_spec["route_key"]
        self.virtual_open = bool(self.protocol_spec["virtual_open"])
        self.transport_prefix = str(self.protocol_spec["transport_prefix"])
        self.evaluation_curriculum_state = None
        if self.virtual_open:
            self.evaluation_curriculum_state = _require_v8_final_evaluation_state(
                compressor_config, runtime_contract.get("evaluation_curriculum_state")
            )
            attached_state = getattr(
                model, "vision_token_compressor_evaluation_curriculum_state", None
            )
            if attached_state != self.evaluation_curriculum_state:
                raise RuntimeError("loaded V8 model lost its final curriculum evaluation binding")
        runtime_identity = {
            "algorithm": self.protocol_spec["algorithm"],
            "method": self.protocol_spec["method"],
            "route_schema_version": self.protocol_spec["route_schema"],
            "transport_key": self.route_key,
        }
        identity_drift = {
            key: {"expected": expected, "actual": compressor_config.get(key)}
            for key, expected in runtime_identity.items()
            if compressor_config.get(key) != expected
        }
        if identity_drift:
            raise RuntimeError(
                f"inference runtime compressor/protocol identity drift: {identity_drift}"
            )
        protocol = self.contract.get("response_protocol")
        if protocol is None:
            protocol = self.contract.get("deployment", {}).get("response_protocol")
        if self.virtual_open:
            if (
                not isinstance(protocol, dict)
                or protocol.get("schema_version") != V8_RESPONSE_PROTOCOL_SCHEMA
                or protocol.get("virtual_open") is not True
                or protocol.get("transport_prefix") != "<answer>"
            ):
                raise RuntimeError("V8 inference runtime lacks its bound virtual-open protocol")
        elif protocol is not None:
            raise RuntimeError("legacy inference runtime unexpectedly carries a V8 response protocol")

        self.export_dir = pathlib.Path(export_dir).resolve()
        self.model = model
        self.processor = processor
        generation_config = align_qwen35_chat_generation_config(
            getattr(model, "generation_config", None),
            tokenizer=processor.tokenizer,
            model_config=model.config,
            required=True,
        )
        expected_eos = [
            int(processor.tokenizer.eos_token_id),
            int(processor.tokenizer.pad_token_id),
        ]
        if (
            list(generation_config.eos_token_id) != expected_eos
            or int(generation_config.pad_token_id) != expected_eos[1]
        ):
            raise RuntimeError(
                "exported Qwen3.5 runtime must stop on <|im_end|> or "
                "<|endoftext|> and pad with <|endoftext|>"
            )
        self.generation_config = generation_config
        original_generate = getattr(model, "_v6_dpc_transformers_generate", None)
        if not callable(original_generate):
            raise RuntimeError("embedded actor did not retain its audited Transformers generation entry point")

        def _functional_dense_or_no_image_generate(*args, **kwargs):
            mode = kwargs.get("visual_compression_mode")
            if mode not in {"dense", "no_image"}:
                raise RuntimeError(
                    "direct Transformers generation is authorized only through the embedded "
                    "runtime with explicit visual_compression_mode='dense' or 'no_image'"
                )
            if kwargs.get("use_cache") is not True:
                raise RuntimeError("audited dense/no-image generation requires the standard KV cache")
            return original_generate(*args, **kwargs)

        model.generate = _functional_dense_or_no_image_generate
        rollout = AuditedHFRollout.__new__(AuditedHFRollout)
        rollout.config = RolloutConfig(
            name="hf",
            mode="async",
            do_sample=False,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            n=1,
            prompt_length=24576,
            response_length=1024,
            hf_dart_decode_batch_size=1,
            dtype="bfloat16",
            hf_use_replicated_module=True,
            hf_preserve_cuda_cache=True,
            semantic_stop_enabled=True,
            semantic_stop_string="</answer>",
            semantic_stop_include_tokens=True,
            semantic_stop_scan_response_only=True,
            semantic_stop_virtual_open=self.virtual_open,
            semantic_stop_transport_prefix=self.transport_prefix,
        )
        rollout.model_config = SimpleNamespace(
            vision_token_compressor=compressor_config,
            image_token_id=int(model.config.image_token_id),
            hf_config=model.config,
        )
        rollout.device_mesh = None
        rollout.module = model
        rollout.keep_module_in_eval = True
        rollout.tokenizer = processor.tokenizer
        rollout.processor = processor
        self.rollout = rollout

    def _make_prompts(
        self,
        messages_batch,
        *,
        response_length,
        visual_compression_mode,
        route_queries_batch=None,
        sample_indices=None,
    ):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto

        if visual_compression_mode not in {"merge", "dense", "no_image"}:
            raise ValueError(
                "visual_compression_mode must be one of ['dense', 'merge', 'no_image'], "
                f"got {visual_compression_mode!r}"
            )
        if not isinstance(messages_batch, list) or not messages_batch:
            raise ValueError("messages_batch must be a non-empty list of per-sample message lists")
        if isinstance(response_length, bool) or not 1 <= int(response_length) <= 1024:
            raise ValueError("response_length must be an integer in [1, 1024]")
        raw = np.empty(len(messages_batch), dtype=object)
        for index, messages in enumerate(messages_batch):
            if not isinstance(messages, list) or not messages:
                raise ValueError("every sample must contain a non-empty message list")
            raw[index] = messages
        non_tensor_batch = {"raw_prompt": raw}
        if self.protocol_spec["algorithm"] == "qwen35_cdpruner_v1":
            normalized = _canonical_v8_route_queries(
                route_queries_batch, len(messages_batch)
            )
            if sample_indices is None:
                normalized_indices = list(range(len(messages_batch)))
            elif (
                not isinstance(sample_indices, list)
                or len(sample_indices) != len(messages_batch)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in sample_indices
                )
                or len(set(sample_indices)) != len(sample_indices)
            ):
                raise ValueError(
                    "V8 sample_indices must be unique, non-negative integers aligned to messages_batch"
                )
            else:
                normalized_indices = list(sample_indices)
            normalized_queries = np.empty(len(normalized), dtype=object)
            rollout_uids = np.empty(len(normalized), dtype=object)
            for index, canonical in enumerate(normalized):
                normalized_queries[index] = canonical
                rendered_prompt = self.processor.apply_chat_template(
                    messages_batch[index],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rollout_uids[index] = _canonical_v8_rollout_uid(
                    rendered_prompt,
                    canonical,
                    normalized_indices[index],
                )
            non_tensor_batch["route_query"] = normalized_queries
            non_tensor_batch["uid"] = rollout_uids
        elif route_queries_batch is not None:
            raise ValueError("legacy V6/V7 query-independent DPC does not accept route_queries_batch")
        elif sample_indices is not None:
            raise ValueError("legacy V6/V7 DPC does not accept V8 sample_indices")
        return DataProto(
            batch=TensorDict(
                {"dummy_tensor": torch.zeros((len(messages_batch), 1), dtype=torch.float32)},
                batch_size=[len(messages_batch)],
            ),
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "do_sample": False,
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
                "response_length": int(response_length),
                "eos_token_id": list(self.generation_config.eos_token_id),
                "pad_token_id": int(self.generation_config.pad_token_id),
                "visual_compression_mode": visual_compression_mode,
            },
        )

    def _capture_physical_prefill(self, *, require_pixels=True):
        capture = {}

        def hook(module, args, kwargs, output):
            del module, args
            if capture or (require_pixels and kwargs.get("pixel_values") is None):
                return
            attention = getattr(output, "dart_attention_mask", None)
            position = getattr(output, "dart_position_ids", None)
            if attention is not None and position is not None:
                capture["attention_mask"] = attention.detach().cpu()
                capture["position_ids"] = position.detach().cpu()

        try:
            handle = self.model.register_forward_hook(hook, with_kwargs=True)
        except TypeError as exc:  # pragma: no cover - pinned torch supports kwargs hooks
            raise RuntimeError("pinned PyTorch must support with_kwargs forward hooks") from exc
        return capture, handle

    def _score_single_token_ids(
        self,
        messages,
        target_token_ids,
        *,
        visual_compression_mode,
        route_query=None,
        sample_index=0,
    ):
        import contextlib
        import math
        import torch
        from verl.utils.model import extract_multi_modal_inputs

        if not isinstance(target_token_ids, (list, tuple)) or not target_token_ids:
            raise ValueError("every scored continuation must contain at least one token ID")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in target_token_ids):
            raise ValueError("scored continuation token IDs must be non-negative integers")
        prompts = self._make_prompts(
            [messages],
            response_length=min(1024, max(1, len(target_token_ids))),
            visual_compression_mode=visual_compression_mode,
            route_queries_batch=([route_query] if route_query is not None else None),
            sample_indices=(
                [sample_index]
                if self.protocol_spec["algorithm"] == "qwen35_cdpruner_v1"
                else None
            ),
        )
        (
            prompt_ids,
            prompt_attention,
            prompt_positions,
            compression_query_mask,
            compression_query_audits,
            _eos_token_id,
            _pad_token_id,
        ) = self.rollout._extract_prompt_tensors(prompts)
        if prompt_ids.shape[0] != 1 or not bool(prompt_attention.to(torch.bool).all().item()):
            raise RuntimeError("single-sample reference scoring requires one unpadded logical prompt")
        if self.virtual_open:
            rendered_prompt = self.processor.tokenizer.decode(
                prompt_ids[0].detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not rendered_prompt.endswith(self.transport_prefix):
                raise RuntimeError("V8 scored prompt does not end in the bound virtual-open prefix")
        device = prompt_ids.device
        target = torch.tensor([list(target_token_ids)], dtype=prompt_ids.dtype, device=device)
        target_attention = torch.ones_like(target, dtype=prompt_attention.dtype)
        input_ids = torch.cat((prompt_ids, target), dim=-1)
        attention_mask = torch.cat((prompt_attention, target_attention), dim=-1)
        model_query_kwargs = {}
        full_query_mask = None
        if visual_compression_mode == "merge" and self.protocol_spec["algorithm"] == "qwen35_cdpruner_v1":
            if compression_query_mask is None or compression_query_mask.shape != prompt_ids.shape:
                raise RuntimeError("CDPruner reference scoring query-mask shape drift")
            full_query_mask = _extend_prompt_query_mask(compression_query_mask, target)
            if full_query_mask.shape != input_ids.shape or bool(
                full_query_mask[:, -target.shape[-1] :].any().item()
            ):
                raise RuntimeError("reference target tokens leaked into the CDPruner query mask")
            model_query_kwargs["compression_query_mask"] = full_query_mask
        elif (
            visual_compression_mode == "merge"
            and compression_query_mask is not None
            and bool(compression_query_mask.any().item())
        ):
            raise RuntimeError("legacy query-independent DPC unexpectedly received query tokens")
        delta = torch.arange(1, target.shape[-1] + 1, device=device, dtype=prompt_positions.dtype)
        if prompt_positions.dim() == 3:
            response_positions = prompt_positions[..., -1:] + delta.view(1, 1, -1)
            position_ids = torch.cat((prompt_positions, response_positions), dim=-1)
            model_position_ids = position_ids.transpose(0, 1)
        else:
            response_positions = prompt_positions[:, -1:] + delta.view(1, -1)
            position_ids = torch.cat((prompt_positions, response_positions), dim=-1)
            model_position_ids = position_ids

        multi_modal_inputs = extract_multi_modal_inputs(
            prompts.non_tensor_batch["multi_modal_inputs"]
        )
        multi_modal_inputs.pop("images_seqlens", None)
        multi_modal_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in multi_modal_inputs.items()
        }
        if visual_compression_mode == "no_image":
            model_multi_modal_inputs = {}
        else:
            if not multi_modal_inputs:
                raise RuntimeError(f"{visual_compression_mode} scoring requires one image input")
            model_multi_modal_inputs = multi_modal_inputs
        if visual_compression_mode == "merge":
            self.rollout._clear_and_collect_dart_routes(clear=True)

        parameter = next(self.model.parameters())
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if parameter.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=model_position_ids,
                visual_compression_mode=visual_compression_mode,
                logits_to_keep=0,
                use_cache=False,
                return_dict=True,
                **model_query_kwargs,
                **model_multi_modal_inputs,
            )
        logits = output.logits
        target_length = int(target.shape[-1])
        physical_prompt_length = int(logits.shape[1]) - target_length
        if physical_prompt_length <= 0:
            raise RuntimeError("reference scoring produced no physical prompt prefix")
        prediction_positions = torch.arange(
            physical_prompt_length - 1,
            physical_prompt_length + target_length - 1,
            device=logits.device,
            dtype=torch.long,
        )
        selected_logits = logits[0].index_select(0, prediction_positions).float()
        token_log_probs = (
            selected_logits.gather(-1, target[0].long().unsqueeze(-1)).squeeze(-1)
            - torch.logsumexp(selected_logits, dim=-1)
        )
        if not bool(torch.isfinite(token_log_probs).all().item()):
            raise RuntimeError("reference scoring produced a non-finite token log-probability")
        routes = []
        if visual_compression_mode == "merge":
            routes = self.rollout._clear_and_collect_dart_routes(clear=False)
            if not routes:
                raise RuntimeError("merge reference scoring did not emit a DPC route")
            if self.protocol_spec["algorithm"] == "qwen35_cdpruner_v1":
                if (
                    not isinstance(compression_query_audits, list)
                    or len(compression_query_audits) != 1
                    or not isinstance(compression_query_audits[0], dict)
                ):
                    raise RuntimeError("CDPruner reference scoring lost its query audit")
                for route in routes:
                    route["query_audit"] = json.loads(
                        json.dumps(compression_query_audits[0], ensure_ascii=False)
                    )
                routes = _bind_route_identity(
                    [routes],
                    self.protocol_spec,
                    evaluation_curriculum_state=self.evaluation_curriculum_state,
                )[0]
                routes = _validate_v8_route_query_audits(
                    [routes],
                    [route_query],
                    prompts.non_tensor_batch["uid"],
                )[0]
        physical_attention = getattr(output, "dart_attention_mask", None)
        physical_positions = getattr(output, "dart_position_ids", None)
        if physical_attention is None or physical_positions is None:
            raise RuntimeError("reference scoring did not expose physical attention/M-RoPE tensors")
        values = token_log_probs.detach().cpu().tolist()
        nll = -math.fsum(float(value) for value in values) / len(values)
        result = {
            "visual_compression_mode": visual_compression_mode,
            "prompt_token_ids": prompt_ids[0].detach().cpu().tolist(),
            "target_token_ids": list(target_token_ids),
            "token_log_probs": values,
            "nll": nll,
            self.route_key: routes,
            "logical_position_ids": position_ids.detach().cpu(),
            "logical_attention_mask": attention_mask.detach().cpu(),
            "physical_position_ids": physical_positions.detach().cpu(),
            "physical_attention_mask": physical_attention.detach().cpu(),
        }
        if self.virtual_open:
            # This evidence is a V8-only addition.  Omitting the key entirely
            # on V6/V7 preserves their historical scorer output contract.
            result["compression_query_mask"] = (
                full_query_mask.detach().cpu() if full_query_mask is not None else None
            )
        return result

    def score_token_ids(
        self,
        messages_batch,
        target_token_ids_batch,
        *,
        visual_compression_mode="merge",
        route_queries_batch=None,
    ):
        if not isinstance(messages_batch, list) or not isinstance(target_token_ids_batch, list):
            raise ValueError("messages_batch and target_token_ids_batch must be lists")
        if not messages_batch or len(messages_batch) != len(target_token_ids_batch):
            raise ValueError("scoring requires equally sized non-empty messages and token-ID batches")
        if route_queries_batch is None:
            route_queries = [None] * len(messages_batch)
        elif not isinstance(route_queries_batch, list) or len(route_queries_batch) != len(messages_batch):
            raise ValueError("route_queries_batch must be batch-aligned with scoring messages")
        else:
            route_queries = route_queries_batch
        samples = [
            self._score_single_token_ids(
                messages,
                target_ids,
                visual_compression_mode=visual_compression_mode,
                route_query=route_query,
                sample_index=sample_index,
            )
            for sample_index, (messages, target_ids, route_query) in enumerate(
                zip(messages_batch, target_token_ids_batch, route_queries, strict=True)
            )
        ]
        return {
            "visual_compression_mode": visual_compression_mode,
            "token_log_probs": [sample["token_log_probs"] for sample in samples],
            "nll": [sample["nll"] for sample in samples],
            "samples": samples,
        }

    def score_reference(
        self,
        messages_batch,
        reference_texts,
        *,
        visual_compression_mode="merge",
        route_queries_batch=None,
    ):
        if not isinstance(reference_texts, list) or not reference_texts:
            raise ValueError("reference_texts must be a non-empty list")
        target_ids = []
        for reference in reference_texts:
            suffix = _reference_target_text(reference, self.protocol_spec)
            encoded = self.processor.tokenizer(
                suffix,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if not encoded:
                raise RuntimeError("reference answer produced an empty assistant continuation")
            if len(encoded) > 1024:
                raise ValueError(
                    "reference answer exceeds the audited 1024-token continuation limit"
                )
            encoded = [int(value) for value in encoded]
            if self.virtual_open:
                forbidden_ids = {
                    int(value)
                    for value in getattr(self.processor.tokenizer, "all_special_ids", [])
                }
                forbidden_ids.update(
                    int(value)
                    for value in (
                        self.processor.tokenizer.eos_token_id,
                        self.processor.tokenizer.pad_token_id,
                    )
                    if value is not None
                )
                if any(value in forbidden_ids for value in encoded):
                    raise RuntimeError("V8 reference target unexpectedly contains EOS/pad tokens")
            target_ids.append(encoded)
        result = self.score_token_ids(
            messages_batch,
            target_ids,
            visual_compression_mode=visual_compression_mode,
            route_queries_batch=route_queries_batch,
        )
        result["reference_target_policy"] = self.protocol_spec["reference_target_policy"]
        result["reference_text_sha256"] = [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in reference_texts
        ]
        return result

    def generate(
        self,
        messages_batch,
        *,
        response_length=1024,
        visual_compression_mode="merge",
        route_queries_batch=None,
        return_audit=False,
        audit_level=None,
    ):
        import torch

        if audit_level is None:
            audit_level = "full" if return_audit else "none"
        if audit_level not in {"none", "route", "full"}:
            raise ValueError("audit_level must be one of ['full', 'none', 'route']")
        if return_audit and audit_level != "full":
            raise ValueError("return_audit=True is compatible only with audit_level='full'")
        # Full parity capture transfers one complete 248k-vocabulary logit
        # vector to CPU for every decode token.  It is an audit primitive, not
        # a production-generation mode; fail closed before allocating hundreds
        # of MiB/GiB when a caller accidentally requests a long answer or batch.
        if audit_level == "full" and (
            int(response_length) > 2
            or not isinstance(messages_batch, list)
            or len(messages_batch) != 1
        ):
            raise ValueError(
                "audit_level='full' is restricted to one sample and at most two decode tokens; "
                "use audit_level='route' or 'none' for evaluation"
            )
        prompts = self._make_prompts(
            messages_batch,
            response_length=response_length,
            visual_compression_mode=visual_compression_mode,
            route_queries_batch=route_queries_batch,
        )
        self.rollout._v6_capture_level = audit_level
        self.rollout._v6_decode_logits = []
        self.rollout._v6_greedy_tokens = []
        if audit_level == "full":
            physical, handle = self._capture_physical_prefill(
                require_pixels=visual_compression_mode != "no_image"
            )
        else:
            physical, handle = {}, None
        try:
            with torch.inference_mode():
                output = self.rollout.generate_sequences(prompts)
        finally:
            if handle is not None:
                handle.remove()
            self.rollout._v6_capture_level = "none"
        responses = output.batch["responses"].detach()
        response_mask = output.batch.get("response_mask")
        if response_mask is None or response_mask.shape != responses.shape:
            raise RuntimeError("embedded rollout did not return an explicit response action mask")
        response_mask = response_mask.to(dtype=torch.long)
        if not torch.equal(response_mask, response_mask.to(torch.bool).to(torch.long)):
            raise RuntimeError("embedded rollout returned a non-binary response action mask")
        if bool((response_mask[:, 1:] > response_mask[:, :-1]).any().item()):
            raise RuntimeError("embedded rollout response action mask is not prefix-contiguous")
        if not torch.equal(output.batch["attention_mask"][:, -responses.shape[-1] :], response_mask):
            raise RuntimeError("embedded rollout attention suffix differs from its response action mask")
        rollout_log_probs = output.batch.get("rollout_log_probs")
        if rollout_log_probs is None:
            scored = self.score_token_ids(
                messages_batch,
                [
                    row[mask.to(torch.bool)].detach().cpu().tolist()
                    for row, mask in zip(responses, response_mask, strict=True)
                ],
                visual_compression_mode=visual_compression_mode,
                route_queries_batch=route_queries_batch,
            )
            rollout_log_probs = torch.zeros_like(responses, dtype=torch.float32)
            for index, token_log_probs in enumerate(scored["token_log_probs"]):
                length = len(token_log_probs)
                rollout_log_probs[index, :length] = torch.as_tensor(
                    token_log_probs,
                    device=rollout_log_probs.device,
                    dtype=torch.float32,
                )
        if self.virtual_open:
            from verl.utils.response_protocol import classify_answer_protocol

            raw_continuations = _decode_masked_response_rows(
                self.processor.tokenizer, responses, response_mask
            )
            classified = [
                classify_answer_protocol(
                    value,
                    virtual_open=True,
                    transport_prefix=self.transport_prefix,
                )
                for value in raw_continuations
            ]
            protocol_predictions = [item.reconstructed_text for item in classified]
            protocol_status = [item.status for item in classified]
            rollout_status = output.non_tensor_batch.get("rollout_protocol_status")
            rollout_prefix = output.non_tensor_batch.get("rollout_response_transport_prefix")
            rollout_reason = output.non_tensor_batch.get("rollout_stop_reason")
            if (
                rollout_status is None
                or rollout_prefix is None
                or list(rollout_prefix) != [self.transport_prefix] * len(messages_batch)
                or rollout_reason is None
                or len(rollout_reason) != len(messages_batch)
            ):
                raise RuntimeError("V8 rollout protocol/status/prefix evidence disagrees with masked decode")
            rollout_status = list(rollout_status)
            rollout_reason = list(rollout_reason)
            for semantic_status, strict_status, stop_reason in zip(
                rollout_status, protocol_status, rollout_reason, strict=True
            ):
                if semantic_status == "not_closed":
                    if strict_status not in {"missing_close", "nested_open"}:
                        raise RuntimeError(
                            "V8 strict protocol reports a close-only state while semantic stop reports no close"
                        )
                    if stop_reason not in {"length", "token_eos"}:
                        raise RuntimeError("V8 unclosed response has an incompatible rollout stop reason")
                elif semantic_status != strict_status or stop_reason != "answer_tag":
                    raise RuntimeError(
                        "V8 semantic-stop status/reason disagrees with strict protocol status"
                    )
            decoded_predictions = protocol_predictions
        else:
            # Preserve the historical V6/V7 decoder exactly for golden
            # compatibility.  Only V8 consumes the stricter masked decoder.
            decoded_predictions = self.processor.tokenizer.batch_decode(
                responses.detach().cpu(), skip_special_tokens=True
            )
        result = {
            "visual_compression_mode": visual_compression_mode,
            "responses": responses.cpu(),
            "response_mask": response_mask.cpu(),
            "rollout_log_probs": rollout_log_probs.detach().cpu(),
            "decoded_predictions": decoded_predictions,
        }
        if self.virtual_open:
            result.update(
                {
                    "decoded_raw_continuations": raw_continuations,
                    "decoded_protocol_predictions": protocol_predictions,
                    "protocol_status": protocol_status,
                    "semantic_stop_protocol_status": rollout_status,
                    "rollout_stop_reason": rollout_reason,
                    "response_transport_prefix": [self.transport_prefix] * len(messages_batch),
                }
            )
        emitted_routes = output.non_tensor_batch.get(self.route_key)
        if self.virtual_open and visual_compression_mode == "merge":
            if emitted_routes is None:
                raise RuntimeError("V8 merge rollout omitted its CDPruner route transport")
            emitted_routes = _bind_route_identity(
                emitted_routes,
                self.protocol_spec,
                evaluation_curriculum_state=self.evaluation_curriculum_state,
            )
            emitted_routes = _validate_v8_route_query_audits(
                emitted_routes,
                route_queries_batch,
                prompts.non_tensor_batch["uid"],
            )
        elif self.virtual_open and emitted_routes is not None:
            raise RuntimeError("V8 dense/no-image rollout unexpectedly emitted CDPruner routes")
        if audit_level in {"route", "full"}:
            if emitted_routes is None:
                emitted_routes = [[] for _ in range(len(messages_batch))]
            result[self.route_key] = emitted_routes
        if audit_level == "full":
            if set(physical) != {"attention_mask", "position_ids"}:
                raise RuntimeError("embedded runtime failed to capture the physical DPC prefill tensors")
            result.update(
                {
                    "logical_position_ids": output.batch["position_ids"].detach().cpu(),
                    "logical_attention_mask": output.batch["attention_mask"].detach().cpu(),
                    "physical_prefill_position_ids": physical["position_ids"],
                    "physical_prefill_attention_mask": physical["attention_mask"],
                    "decode_logits": list(self.rollout._v6_decode_logits),
                    "greedy_token_ids": list(self.rollout._v6_greedy_tokens),
                }
            )
        return result


def _load_v6_dpc_runtime_from_validated_contract(
    export_dir, contract, *, torch_dtype="auto", device_map="auto"
):
    model = _load_validated_v6_dpc_model(
        export_dir,
        contract,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5Model

    model_dir = export_dir / "compressed_actor_model"
    raw_config = json.loads((model_dir / "v6_dpc_config.json").read_text(encoding="utf-8"))
    model_type = raw_config.pop("model_type")
    model_config = AutoConfig.for_model(model_type, **raw_config)
    processor = AutoProcessor.from_pretrained(model_dir, config=model_config, trust_remote_code=True)
    if processor.__class__.__name__ != "Qwen3VLProcessor":
        raise RuntimeError(f"unexpected processor type: {processor.__class__.__name__}")
    resize = contract["processor_resize"]
    size = getattr(processor.image_processor, "size", None)
    expected_size = {"shortest_edge": resize["min_pixels"], "longest_edge": resize["max_pixels"]}
    if size != expected_size:
        raise RuntimeError(f"exported processor resize drift: expected={expected_size}, actual={size}")
    processor.config = model_config
    processor.get_rope_index = MethodType(Qwen3_5Model.get_rope_index, processor)
    if not hasattr(Qwen3_5Model, "get_vision_position_ids"):
        raise RuntimeError("Qwen3.5 runtime is missing get_vision_position_ids")
    processor.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, processor)
    template = (export_dir / "chat_template.jinja").read_text(encoding="utf-8")
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    return V6DPCInferenceRuntime(export_dir, model, processor, contract)


def load_v6_dpc_runtime(export_dir, *, torch_dtype="auto", device_map="auto"):
    """Load a deployment artifact; diagnostic-only exports fail closed here."""

    export_dir, contract = _load_and_validate_contract(export_dir, consumer="deployment")
    return _load_v6_dpc_runtime_from_validated_contract(
        export_dir,
        contract,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )






__all__ = [
    "V6DPCInferenceRuntime",
    "load_v6_dpc_model",
    "load_v6_dpc_runtime",
    "prepare_loaded_model_for_v6_dpc_runtime",
    "_decode_masked_response_rows",
    "_canonical_v8_route_queries",
    "_canonical_v8_rollout_uid",
    "_validate_v8_route_query_audits",
    "_extend_prompt_query_mask",
    "_reference_target_text",
]


load_cdpruner_runtime = load_v6_dpc_runtime
CDPrunerInferenceRuntime = V6DPCInferenceRuntime
