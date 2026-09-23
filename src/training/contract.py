"""Portable V8 configuration and checkpoint identities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def load_contract(path=None):
    source = Path(path) if path else ROOT / 'v8.json'
    raw = source.read_bytes()
    value = json.loads(raw)
    return value, {'schema_version': value['schema_version'],
                   'release_variant': value['release_variant'],
                   'file_sha256': hashlib.sha256(raw).hexdigest(),
                   'canonical_sha256': canonical_sha256(value)}


def load_runtime_optimization(path=None):
    source = Path(path) if path else ROOT / 'runtime.json'
    value = json.loads(source.read_text())
    value['base_static_contract'] = load_contract()[1]
    return value, {'schema_version': value['schema_version'], 'name': value['name'],
                   'file_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'canonical_sha256': canonical_sha256(value),
                   'base_static_contract': value['base_static_contract']}


def effective_runtime_settings(runtime, profile):
    return copy.deepcopy(runtime['effective_profiles'][profile])


def scientific_runtime_identity(contract):
    training = contract['training']
    return {
        'distillation': {'alpha': float(training['distillation_alpha']),
                         'gamma': float(training['distillation_gamma']),
                         'importance_sampling_clip': float(training['importance_sampling_clip'])},
        'rollout_sampling': copy.deepcopy(contract['rollout']['sampling']),
        'actor_model_initialization_seed': int(contract['actor']['model_initialization_seed']),
        'visual_token_curriculum': copy.deepcopy(contract['compressor']['curriculum']),
    }
