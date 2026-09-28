"""Toy model of PyTorch's CUDA caching allocator (large pool, expandable
segments) replaying the patch_v13_fp8 weight swap on a per-rank TP=2 layout.

Why: after the swap, torch reserved memory fell by only part of the freed
bytes (E2a 1.52 of 3.42 GiB). This model reproduces E2a/E2b/E2c/E2d to within
~0.3 GiB and ranks the fix candidates. It is a model, not a measurement: the
serve log line "torch reserved X -> Y GiB" is the check.

Allocator rules modelled (c10/cuda/CUDACachingAllocator.cpp):
  - run.sh sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: one virtual
    range per stream, physical pages of kLargeBuffer = 20 MiB
  - vLLM's load_model runs under max_split_size_mb:20 (gpu_worker.py
    _scoped_allocator_max_split): a request < 20 MiB may not take a free block
    >= 20 MiB, and a larger request may not take one >= request + 20 MiB
  - malloc: best-fit over free *mapped* blocks, else the lowest-address
    unmapped block (with a free mapped block just before it) that has room,
    mapping whole pages; split if the remainder is > 1 MiB
  - free: merge with free mapped neighbours
  - empty_cache: unmap every whole page inside a free mapped block
  - reserved = mapped pages * 20 MiB; sizes <= 1 MiB (small pool) are ignored
Layout: module order of Glm5Next (vision, embed, 45 layers of attention, router,
shared experts, experts, mHC), lm_head, then the MoE Marlin repack with an
empty_cache per layer (release_device_memory_under_pressure).

  python3 allocsim.py [case ...]      # MAX_SPLIT_MB=0 models an unlimited split
"""

import bisect
import os
import sys

P = 20 << 20
MAX_SPLIT = int(os.environ.get("MAX_SPLIT_MB", "20")) << 20  # 0 = unlimited
SMALL = 1 << 20
MiB = 2**20


class Sim:
    def __init__(self):
        self.starts = [0]
        self.blocks = {0: [0, 1 << 50, "U"]}
        self.nh = 0
        self.h = {}

    def _prev(self, s):
        i = bisect.bisect_left(self.starts, s)
        return self.blocks[self.starts[i - 1]] if i > 0 else None

    def _next(self, b):
        i = bisect.bisect_right(self.starts, b[0])
        return self.blocks[self.starts[i]] if i < len(self.starts) else None

    def _add(self, s, size, st):
        bisect.insort(self.starts, s)
        self.blocks[s] = [s, size, st]
        return self.blocks[s]

    def _rm(self, b):
        self.starts.pop(bisect.bisect_left(self.starts, b[0]))
        del self.blocks[b[0]]

    def _merge(self, b):
        n = self._next(b)
        if n and n[2] == b[2]:
            b[1] += n[1]
            self._rm(n)
        p = self._prev(b[0])
        if p and p[2] == b[2]:
            p[1] += b[1]
            self._rm(b)
            b = p
        return b

    def malloc(self, size):
        if size <= SMALL:
            return None
        size = -(-size // 512) * 512
        best = None
        for b in self.blocks.values():
            if b[2] == "F" and b[1] >= size and (best is None or (b[1], b[0]) < (best[1], best[0])):
                best = b
        # get_free_block's oversize rules (vLLM loads under max_split_size_mb:20)
        if best is not None and MAX_SPLIT and (
            (size < MAX_SPLIT and best[1] >= MAX_SPLIT)
            or (size >= MAX_SPLIT and best[1] >= size + P)
        ):
            best = None
        if best is None:
            for s in self.starts:
                u = self.blocks[s]
                if u[2] != "U":
                    continue
                c = u
                p = self._prev(u[0])
                if p and p[2] == "F":
                    c = p
                room, x = 0, c
                while x and x[2] != "A" and room < size:
                    room += x[1]
                    x = self._next(x)
                if room >= size:
                    break
            # map pages covering [c.start, c.start + size)
            c0 = c[0]
            end = -(-(c0 + size) // P) * P
            x = c
            while x and x[0] < c0 + size:
                assert x[2] != "A"
                if x[2] == "U":
                    xe = x[0] + x[1]
                    if xe > end:
                        self._add(end, xe - end, "U")
                        x[1] = end - x[0]
                    x[2] = "F"
                    x = self._merge(x)
                x = self._next(x)
            best = self._prev(c0 + 1)
            assert best[2] == "F" and best[1] >= size, best
        s0 = best[0]
        rem = best[1] - size
        if rem > SMALL:
            best[1] = size
            self._add(s0 + size, rem, "F")
        best[2] = "A"
        self.nh += 1
        self.h[self.nh] = s0
        return self.nh

    def free(self, h):
        if h is None:
            return
        b = self.blocks[self.h.pop(h)]
        b[2] = "F"
        self._merge(b)

    def empty_cache(self):
        for s in list(self.starts):
            b = self.blocks.get(s)
            if not b or b[2] != "F":
                continue
            s0, e0 = b[0], b[0] + b[1]
            ps, pe = -(-s0 // P) * P, e0 // P * P
            if pe <= ps:
                continue
            self._rm(b)
            if ps > s0:
                self._add(s0, ps - s0, "F")
            u = self._add(ps, pe - ps, "U")
            if e0 > pe:
                self._add(pe, e0 - pe, "F")
            self._merge(u)

    def reserved(self):
        pages = set()
        for b in self.blocks.values():
            if b[2] != "U":
                pages.update(range(b[0] // P, (b[0] + b[1] - 1) // P + 1))
        return len(pages) * P

    def allocated(self):
        return sum(b[1] for b in self.blocks.values() if b[2] == "A")


# ---------------------------------------------------------------- model layout
KDA_IN, KDA_O = (12576, 4096), (4096, 4096)
Q_B, MLA_O = (8192, 1536), (4096, 8192)
SH_GU, SH_D = (2048, 4096), (4096, 1024)
LM = (77440, 4096)
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}


def bf16(nk):
    return nk[0] * nk[1] * 2


def build_target(sim, groups):
    """Allocate params in module order; return list of (group, handle, (n, k))."""
    sites, experts = [], []
    sim.malloc(1024 << 20)  # vision tower, roughly
    sim.malloc(bf16(LM))  # embed_tokens
    for L in range(45):
        if L in MLA_LAYERS:
            sim.malloc(2112 * 4096 * 2)  # fused_qkv_a
            sites.append(("mla", sim.malloc(bf16(Q_B)), Q_B))
            sim.malloc(16384 * 512 * 2)  # kv_b
            sites.append(("mla", sim.malloc(bf16(MLA_O)), MLA_O))
            sim.malloc(4096 * 1536 * 2)  # indexer wq_b
            sim.malloc(160 * 4096 * 2)  # wk_weights_proj
        else:
            sites.append(("kda_in", sim.malloc(bf16(KDA_IN)), KDA_IN))
            sites.append(("kda_o", sim.malloc(bf16(KDA_O)), KDA_O))
        if L < 3:
            for sz in (12288 * 2048, 12288 * 256, 4096 * 3072, 4096 * 384):
                sim.malloc(sz)
        else:
            sim.malloc(288 * 4096 * 2)  # router gate
            sites.append(("shared", sim.malloc(bf16(SH_GU)), SH_GU))
            sites.append(("shared", sim.malloc(bf16(SH_D)), SH_D))
            experts.append([sim.malloc(sz) for sz in
                            (1152 * MiB, 144 * MiB, 576 * MiB, 72 * MiB)])
        sim.malloc(24 * 16384 * 4)
        sim.malloc(24 * 16384 * 4)
    sites.append(("lm_head", sim.malloc(bf16(LM)), LM))
    # MoE Marlin repack: new tensors, free old, UMA pressure release per module
    for ex in experts:
        for i, sz in enumerate((1152 * MiB, 144 * MiB, 576 * MiB, 72 * MiB)):
            new = sim.malloc(sz)
            sim.free(ex[i])
            ex[i] = new
        sim.empty_cache()
    for _ in MLA_LAYERS:  # W_UK_T / W_UV
        sim.malloc(2 * MiB + 4096)
        sim.malloc(2 * MiB + 4096)
    return [s for s in sites if s[0] in groups]


def padded_n(n):
    return min((-(-n // 64) * 64, -(-n // 128) * 128))


def swap_layer(sim, h, nk, order, mode):
    """One layer's swap, allocation by allocation, as patch_v13_fp8 does it:
    Q = quantize (outputs, then chunked fp32 transients), D = drop the BF16
    weight, P = prepare_*_for_marlin / repack (transients, final tensors),
    E = torch.cuda.empty_cache(), C = _compact's copies of the final tensors."""
    n, k = nk
    npad = padded_n(n)
    b = {"fp8": 8, "int8": 8, "nvfp4": 4, "int4": 4}[mode]
    w_bytes, wp_bytes = n * k * b // 8, npad * k * b // 8
    s_bytes = {"fp8": 0, "nvfp4": n * k // 16}.get(mode, k // 128 * n * 2)
    sp_bytes = {"fp8": 0, "nvfp4": npad * k // 16}.get(mode, k // 128 * npad * 2)
    rows = max(1, (1 << 24) // k) if mode == "fp8" else max(1, (1 << 22) // k)
    chunk = {"fp8": (4, 4, 4, 1), "nvfp4": (4, 4, 8, 0.5)}.get(mode, (4, 4, 4, 4))
    st = {"bf16": h}

    def quantize():
        st["q"], st["s"] = sim.malloc(w_bytes), sim.malloc(s_bytes)
        a = None
        for i in range(0, n, rows):
            r = min(rows, n - i)
            a2 = sim.malloc(int(chunk[0] * r * k))
            sim.free(a)
            a = a2
            sim.free(sim.malloc(int(chunk[1] * r * k)))
            c = sim.malloc(int(chunk[2] * r * k))
            d = sim.malloc(int(chunk[3] * r * k))
            sim.free(c)
            sim.free(d)
        sim.free(a)

    def drop():
        sim.free(st.pop("bf16"))

    def prepare():
        tmp = [sim.malloc(w_bytes) if mode in ("fp8", "nvfp4") else None,  # .T.contiguous()
               sim.malloc(wp_bytes) if npad != n else None]  # marlin_pad_qweight
        st["out"] = [sim.malloc(wp_bytes)]  # gptq_marlin_repack
        tmp += [sim.malloc(sp_bytes), sim.malloc(sp_bytes)]  # scale pad / permute
        st["out"].append(sim.malloc(sp_bytes))
        for t in tmp + [st.pop("q"), st.pop("s")]:
            sim.free(t)

    def clone():
        for i, (h_old, size) in enumerate(zip(st["out"], (wp_bytes, sp_bytes))):
            st["out"][i] = sim.malloc(size)
            sim.free(h_old)

    ops = {"Q": quantize, "D": drop, "P": prepare, "E": sim.empty_cache, "C": clone}
    for op in order:
        ops[op]()


def run(groups, order, mode):
    sim = Sim()
    sites = build_target(sim, groups)
    r0, a0 = sim.reserved(), sim.allocated()
    for _, h, nk in sites:
        swap_layer(sim, h, nk, order, mode)
    sim.empty_cache()
    r1, a1 = sim.reserved(), sim.allocated()
    return (r0 - r1) / 2**30, (a0 - a1) / 2**30


ORDERS = {
    "E2 patch: Q D P, one E at the end": "QDP",
    "Q D E P (E before the repack)": "QDEP",
    "Q P D E (drop after the repack)": "QPDE",
    "Q E D P": "QEDP",
    "Q D P E C (_compact, this patch)": "QDPEC",
}
ALL = {"kda_in", "kda_o", "mla", "shared", "lm_head"}
CASES = {  # name -> (groups, mode, measured "reserved drop of freed", GiB)
    "E2a fp8 all target groups": (ALL, "fp8", "1.52 of 3.42"),
    "E2b nvfp4 all target groups": (ALL, "nvfp4", "2.56 of 4.66"),
    "E2c fp8 kda_o,mla,shared": ({"kda_o", "mla", "shared"}, "fp8", "0.21 of 1.53"),
    "E2d fp8 kda_in": ({"kda_in"}, "fp8", "0.29 of 1.63"),
    "int8 all target groups": (ALL, "int8", "-"),
    "int4 all target groups": (ALL, "int4", "-"),
}

if __name__ == "__main__":
    print(f"page {P >> 20} MiB, max_split_size {MAX_SPLIT >> 20} MiB (vLLM loads under 20)")
    for case in sys.argv[1:] or list(CASES):
        groups, mode, measured = CASES[case]
        print(f"{case}  (measured on the Sparks: reserved fell {measured} GiB)")
        for label, order in ORDERS.items():
            drop, freed = run(groups, order, mode)
            print(f"  {label:36} reserved -{drop:5.2f} GiB of {freed:4.2f} GiB freed "
                  f"({100 * drop / freed:5.1f}%)")
