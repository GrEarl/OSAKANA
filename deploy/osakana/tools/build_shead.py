"""Shortlist head for the drafter, built from the output head of an F16 GGUF (the dequantized YASAKANI head).
usage: build_shead.py <f16.gguf> <out.ysh> [hidden_dump.bin|- [eval_dump.bin]]
File (little endian): 64-byte header "YSH1", V, H, R, offsets of: proj, ascale, sketch, wscale, w8; then
  proj   R x H f32     z = proj h (rows scaled so that sketch . z approximates W h)
  ascale V f32         score[v] = ascale[v] * (sketch8[v] . z8)   (z8: z as int8 per input row, its scale does not matter for ranking)
  sketch V x R int8    in MMA fragment order: [V/16][step 0..15][lane 0..31][16 bytes] = a0..a3 of mma m16n8k32 for rows g, g+8
  wscale V f32         exact row v = wscale[v] * w8[v]
  w8     V x H int8    row major
Without a hidden-state dump the projection is the top-R principal directions of the head. With one (GGML_YTR_SHEAD_DUMP of the engine:
raw f32 rows of H), it is the rank-R map that minimizes the squared logit error over those states (reduced-rank regression)."""
import struct, sys
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "gguf-py"))
import gguf

R = 512
src, out = sys.argv[1], sys.argv[2]
dump = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
evald = sys.argv[4] if len(sys.argv) > 4 else dump
dev = "cuda"
t = next(t for t in gguf.GGUFReader(src).tensors if t.name == "output.weight")
W = torch.from_numpy(np.asarray(t.data).astype(np.float32)).to(dev)          # [V, H]
V, H = W.shape
assert V % 16 == 0 and H % 512 == 0
G = torch.zeros(H, H, dtype=torch.float64, device=dev)
for i in range(0, V, 16384):
    w = W[i:i + 16384].double(); G += w.T @ w
if dump:
    X = torch.from_numpy(np.fromfile(dump, np.float32).reshape(-1, H)).to(dev).double()
    C = X.T @ X / X.shape[0]
    print("hidden states:", X.shape[0])
    # minimize E |W (I - P) h|^2 over rank-R P: with G = L L^T and A = L^T C^(1/2), P = L^-T U_R U_R^T L^T
    ev, evec = torch.linalg.eigh(C); Ch = evec @ torch.diag(ev.clamp_min(0).sqrt()) @ evec.T
    L = torch.linalg.cholesky(G + 1e-6 * torch.eye(H, dtype=torch.float64, device=dev) * G.diagonal().mean())
    U, S, _ = torch.linalg.svd(L.T @ Ch)
    Ur = U[:, :R]
    proj = (Ur.T @ L.T)                                                      # z = proj h
    Q = torch.linalg.solve_triangular(L.T, Ur, upper=True)                   # L^-T U_R : sketch = W Q
    print("captured logit energy: %.4f" % (S[:R].square().sum() / S.square().sum()).item())
else:
    ev, evec = torch.linalg.eigh(G)
    Q = evec[:, -R:].flip(1)                                                 # [H, R]
    proj = Q.T
    print("captured head energy: %.4f" % (ev[-R:].sum() / ev.sum()).item())
Sk = (W.double() @ Q).float()                                                # [V, R]
asc = Sk.abs().amax(1).clamp_min(1e-12) / 127
S8 = torch.round(Sk / asc[:, None]).clamp(-127, 127).to(torch.int8).cpu().numpy()
wsc = W.abs().amax(1).clamp_min(1e-12) / 127
W8 = torch.round(W / wsc[:, None]).clamp(-127, 127).to(torch.int8).cpu().numpy()
# fragment order: [tile][rr][g][step][half][tig][byte] -> [tile][step][g][tig][half][rr][byte]
frag = S8.reshape(V // 16, 2, 8, 16, 2, 4, 4).transpose(0, 3, 2, 5, 4, 1, 6).reshape(-1)
parts = [proj.float().cpu().numpy().tobytes(), asc.cpu().numpy().astype("<f4").tobytes(), frag.tobytes(),
         wsc.cpu().numpy().astype("<f4").tobytes(), W8.tobytes()]
offs, off = [], 64
for p in parts:
    off = (off + 255) // 256 * 256; offs.append(off); off += len(p)
with open(out, "wb") as f:
    f.write(struct.pack("<4sIII5Q", b"YSH1", V, H, R, *offs).ljust(64, b"\0"))
    for o, p in zip(offs, parts):
        f.seek(o); f.write(p)
print("wrote", out, "%.2f GB" % (off / 1e9), "V", V, "H", H, "R", R)
if evald:   # shortlist recall on dumped states: is the exact top-16 inside the top-512 of the sketch?
    Xf = torch.from_numpy(np.fromfile(evald, np.float32).reshape(-1, H)).to(dev)
    Xf = Xf[torch.randperm(Xf.shape[0], generator=torch.Generator().manual_seed(0))[:4096].to(dev)]
    S8d, pj = torch.from_numpy(S8).to(dev).float(), proj.float()
    h1, h16, n = [0, 0, 0], [0, 0, 0], 0
    for i in range(0, Xf.shape[0], 512):
        x = Xf[i:i + 512]
        ex = (x @ W.T).topk(16, 1).indices
        z = x @ pj.T
        z8 = torch.round(z / (z.abs().amax(1, keepdim=True) / 127))
        sl = ((z8 @ S8d.T) * asc[None, :]).topk(2048, 1).indices
        for mi, M in enumerate((512, 1024, 2048)):
            hit = (ex[:, :, None] == sl[:, None, :M]).any(2).float()
            h1[mi] += hit[:, 0].sum().item(); h16[mi] += hit.mean(1).sum().item()
        n += x.shape[0]
    for mi, M in enumerate((512, 1024, 2048)):
        print("recall of the exact top-1 %.4f, of the top-16 %.4f in the shortlist of %d (%d states of %s)" % (h1[mi] / n, h16[mi] / n, M, n, evald))
