from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh")
s = p.read_text()
old = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;\n"
new = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u) : CTA_TILE_KV;\n"
if s.count(old) != 1:
    raise SystemExit("mla.cuh fp8 tile line match count: %d" % s.count(old))
p.write_text(s.replace(old, new))

p = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py")
s = p.read_text()
old = "            major, minor = get_compute_capability(self.device)\n            if major != 9:\n"
new = "            major, minor = get_compute_capability(self.device)\n            if major not in (9, 12):\n"
if s.count(old) != 1:
    raise SystemExit("_core.py fp8 gate match count: %d" % s.count(old))
p.write_text(s.replace(old, new))
print("fp8 MLA sm12x patches applied")
