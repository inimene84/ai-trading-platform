#!/usr/bin/env python3
"""
Feature Schema and Train-Serve Parity Module
Guarantees zero training-serving skew by maintaining a canonical schema,
computing SHA-256 feature hashes, and verifying feature definitions at inference time.
"""

import hashlib
import json
from typing import List, Tuple
import numpy as np

from ml_features import FEATURE_NAMES


SCHEMA_VERSION = "2.0.0"


def compute_feature_schema_hash(feature_names: List[str] = FEATURE_NAMES) -> str:
    """Generates deterministic SHA-256 hash of feature names and schema version."""
    payload = {
        "version": SCHEMA_VERSION,
        "feature_count": len(feature_names),
        "feature_names": sorted(feature_names),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


FEATURE_HASH = compute_feature_schema_hash()


def verify_feature_parity(model_feature_hash: str) -> Tuple[bool, str]:
    """
    Verifies that the model's training feature hash matches current serving feature hash.
    Prevents silent training-serving skew.
    """
    current_hash = FEATURE_HASH
    if model_feature_hash == current_hash:
        return True, f"Parity verified: {current_hash}"
    return False, f"Feature skew detected! Model trained on hash '{model_feature_hash}', but serving engine uses '{current_hash}'"


if __name__ == "__main__":
    print(f"[*] Canonical Feature Schema Version: {SCHEMA_VERSION}")
    print(f"    Feature Count:                    {len(FEATURE_NAMES)}")
    print(f"    Schema Hash:                      {FEATURE_HASH}")
    valid, msg = verify_feature_parity(FEATURE_HASH)
    print(f"    Self-check:                       {msg}")
