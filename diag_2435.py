"""Diagnosis for vllm-torchtpu#2435: is the decode failure a bkv_sz page-alignment bug?

Prints the default decode/prefill block sizes for the four repro configs, then
re-runs the decode check with explicit block sizes:
  * an aligned bkv_sz (1024) on the failing configs -> expected OK
  * an unaligned bkv_sz (768) on the passing configs -> expected BAD

Run from the vllm-torchtpu checkout on one TPU chip:
  PYTHONPATH=$PWD:$PWD/src TPU_VISIBLE_CHIPS=0 python diag_2435.py
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
VMEM = 64 * 1024 * 1024
LAYOUT = configs.KVLayout.HEAD_ALONG_SUBLANE


def default_blocks(kv_dtype, num_kv_heads):
    model = configs.ModelConfigs(
        num_q_heads=6 * num_kv_heads, num_kv_heads=num_kv_heads, head_dim=HD,
        mask_value=-1e9)
    serve = configs.ServingConfigs(
        num_seqs=8, page_size=PAGE, total_q_tokens=512, num_page_indices=8 * 17,
        dtype_q=jnp.bfloat16, dtype_kv=kv_dtype, dtype_out=jnp.bfloat16,
        kv_layout=LAYOUT)
    return wrapper.calculate_block_sizes(model, serve, VMEM)


def case(kv_dtype, num_kv_heads, kv_len, decode_blocks=None):
    rng = np.random.default_rng(0)
    normal = lambda s, dt: jnp.asarray(rng.standard_normal(s, np.float32)).astype(dt)
    nq = 6 * num_kv_heads
    packing = utils.get_dtype_packing(kv_dtype)
    kvx2 = utils.align_to(2 * num_kv_heads, packing)
    max_tokens, max_seqs = 512, 8
    n_pages = -(-kv_len // PAGE)
    num_pages = n_pages + 8
    q = normal((max_tokens, nq, HD), jnp.bfloat16)
    k = normal((max_tokens, num_kv_heads, HD), kv_dtype)
    v = normal((max_tokens, num_kv_heads, HD), kv_dtype)
    kv = normal((kv_len, kvx2 // packing, packing, HD), kv_dtype)
    kv = jnp.pad(kv, ((0, num_pages * PAGE - kv_len), (0, 0), (0, 0), (0, 0)),
                 constant_values=jnp.nan)
    kv_cache = kv.reshape(num_pages, PAGE, kvx2 // packing, packing, HD)
    pi = np.zeros((max_seqs, n_pages), np.int32)
    pi[0] = np.arange(n_pages)
    page_indices = jnp.asarray(pi.reshape(-1))
    cu = jnp.asarray([0, 1] + [1] * (max_seqs - 1), jnp.int32)
    kv_lens = jnp.asarray([kv_len] + [0] * (max_seqs - 1), jnp.int32)
    dist = jnp.asarray([1, 1, 1], jnp.int32)
    kw = dict(sm_scale=HD**-0.5, sliding_window=None, soft_cap=None,
              q_scale=None, k_scale=None, v_scale=None)
    ref_cache = jnp.nan_to_num(
        kv_cache.astype(jnp.float32).reshape(num_pages, PAGE, kvx2, HD)[:, :, : 2 * num_kv_heads],
        nan=0.0).reshape(num_pages, PAGE, 2 * num_kv_heads, 1, HD)
    exp, _ = ref_ragged_paged_attention(q, k.astype(jnp.float32), v.astype(jnp.float32),
                                        ref_cache, kv_lens, page_indices, cu, dist, **kw)
    out, _ = wrapper.ragged_paged_attention(q, k, v, kv_cache, kv_lens, page_indices, cu, dist,
                                            **kw, vmem_limit_bytes=VMEM, kv_layout=LAYOUT,
                                            decode_block_sizes=decode_blocks)
    o, e = np.asarray(out[:1], np.float32), np.asarray(exp[:1], np.float32)
    return float(np.linalg.norm(o - e) / np.linalg.norm(e))


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print("device:", jax.devices()[0].device_kind, "| vllm-torchtpu @", sha,
          "| jax", jax.__version__)
    cfgs = [(jnp.bfloat16, "bf16", 1), (jnp.bfloat16, "bf16", 2),
            (jnp.float8_e4m3fn, "e4m3", 1), (jnp.float8_e4m3fn, "e4m3", 2)]

    print("\n--- 1. default block sizes (page_size=512) ---")
    for dt, name, h in cfgs:
        d, p = default_blocks(dt, h)
        print(f"{name} kv_heads={h}: decode bkv_sz={d.bkv_sz} (page multiple: "
              f"{d.bkv_sz % PAGE == 0}) batch={d.batch_size} bq={d.bq_sz} | "
              f"prefill bkv_sz={p.bkv_sz} (page multiple: {p.bkv_sz % PAGE == 0}) "
              f"batch={p.batch_size} bq={p.bq_sz}")

    def run(label, blocks):
        print(f"\n--- {label} ---")
        for dt, name, h in cfgs:
            for kv_len in (4096, 4100):
                rel = case(dt, h, kv_len, decode_blocks=blocks)
                tag = "OK" if rel < 0.05 else "BAD"
                print(f"{name} kv_heads={h} kv_len={kv_len}: rel_err={rel:.4f} {tag}")

    run("2. default block sizes", None)
    run("3. explicit decode bkv_sz=1024 (page multiple)",
        configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=1024, batch_size=8))
    run("4. explicit decode bkv_sz=768 (NOT a page multiple)",
        configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=768, batch_size=8))


if __name__ == "__main__":
    main()
