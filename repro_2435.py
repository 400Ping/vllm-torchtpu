"""Verification for vllm-torchtpu#2435.

Batched RPA decode, HEAD_ALONG_SUBLANE: wrong output when 2*num_kv_heads != kv_packing.
One decode token per case against an f32 reference. Page 512, head_dim 128,
6 query heads per KV head. kv_len 4096 is page-aligned; kv_len 4100 has a
NaN-filled page tail, which a correct kernel never reads.

Run from the vllm-torchtpu checkout on one TPU chip:
  PYTHONPATH=$PWD:$PWD/src TPU_VISIBLE_CHIPS=0 python repro_2435.py
"""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tests.kernels.experimental.batched_rpa.correctness_test import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils, wrapper

PAGE, HD = 512, 128


def case(kv_dtype, num_kv_heads, kv_len, mask_invalid_v=None,
         layout=configs.KVLayout.HEAD_ALONG_SUBLANE):
    rng = np.random.default_rng(0)
    normal = lambda s, dt: jnp.asarray(rng.standard_normal(s, np.float32)).astype(dt)
    nq = 6 * num_kv_heads
    packing = utils.get_dtype_packing(kv_dtype)
    kvx2 = utils.align_to(2 * num_kv_heads, packing)  # padded K/V slots per token
    max_tokens, max_seqs = 512, 8
    n_pages = -(-kv_len // PAGE)
    num_pages = n_pages + 8

    q = normal((max_tokens, nq, HD), jnp.bfloat16)
    k = normal((max_tokens, num_kv_heads, HD), kv_dtype)
    v = normal((max_tokens, num_kv_heads, HD), kv_dtype)
    # Pool as vLLM allocates it: [pages, page, kvx2 // packing, packing, HD].
    kv = normal((kv_len, kvx2 // packing, packing, HD), kv_dtype)
    kv = jnp.pad(kv, ((0, num_pages * PAGE - kv_len), (0, 0), (0, 0), (0, 0)),
                 constant_values=jnp.nan)
    kv_cache = kv.reshape(num_pages, PAGE, kvx2 // packing, packing, HD)

    pi = np.zeros((max_seqs, n_pages), np.int32)
    pi[0] = np.arange(n_pages)
    page_indices = jnp.asarray(pi.reshape(-1))
    cu = jnp.asarray([0, 1] + [1] * (max_seqs - 1), jnp.int32)
    kv_lens = jnp.asarray([kv_len] + [0] * (max_seqs - 1), jnp.int32)
    dist = jnp.asarray([1, 1, 1], jnp.int32)  # one decode sequence
    kw = dict(sm_scale=HD**-0.5, sliding_window=None, soft_cap=None,
              q_scale=None, k_scale=None, v_scale=None)

    # f32 reference on the real K/V slots only (no packing padding, no NaN).
    ref_cache = jnp.nan_to_num(
        kv_cache.astype(jnp.float32).reshape(num_pages, PAGE, kvx2, HD)[:, :, : 2 * num_kv_heads],
        nan=0.0,
    ).reshape(num_pages, PAGE, 2 * num_kv_heads, 1, HD)
    exp, _ = ref_ragged_paged_attention(
        q, k.astype(jnp.float32), v.astype(jnp.float32), ref_cache,
        kv_lens, page_indices, cu, dist, **kw)

    out, _ = wrapper.ragged_paged_attention(
        q, k, v, kv_cache, kv_lens, page_indices, cu, dist, **kw,
        vmem_limit_bytes=64 * 1024 * 1024, kv_layout=layout,
        mask_invalid_v=mask_invalid_v)
    o, e = np.asarray(out[:1], np.float32), np.asarray(exp[:1], np.float32)
    return float(np.linalg.norm(o - e) / np.linalg.norm(e))


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print("device:", jax.devices()[0].device_kind, "| vllm-torchtpu @", sha,
          "| jax", jax.__version__)
    for mask in (None, True):
        print(f"\n--- HEAD_ALONG_SUBLANE, mask_invalid_v={mask} ---")
        for dt, name in ((jnp.bfloat16, "bf16"), (jnp.float8_e4m3fn, "e4m3")):
            for h in (1, 2):
                packing = utils.get_dtype_packing(dt)
                for kv_len in (4096, 4100):
                    rel = case(dt, h, kv_len, mask_invalid_v=mask)
                    tag = "OK" if rel < 0.05 else "BAD"
                    print(f"{name} kv_heads={h} (2*heads={2*h}, packing={packing}) "
                          f"kv_len={kv_len}: rel_err={rel:.4f} {tag}")


if __name__ == "__main__":
    main()
