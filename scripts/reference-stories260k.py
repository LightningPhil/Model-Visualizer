"""Independent NumPy full-prefix reference for the bundled trained checkpoint.

Optional verification tool (Python + NumPy), not an application dependency.
The application uses TypeScript and a retained KV cache. This reference reads
the original output-by-input file matrices and recomputes the complete causal
prefix for every output, using NumPy FP32 contractions. It intentionally does
not import the application runtime. Formula/layout source:
https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'examples/stories260k.checkpoint.json').read_text())
folder = root / 'public/models/stories260k'
blob = (folder / 'stories260K.bin').read_bytes()
assert hashlib.sha256(blob).hexdigest() == manifest['weights']['sha256']
D, F, L, H, KV, V, C = struct.unpack('<7i', blob[:28])
HD = D // H
flat = np.frombuffer(blob, '<f4', offset=28)
at = 0
def take(shape):
    global at
    n = int(np.prod(shape))
    data = flat[at:at+n].reshape(shape)
    at += n
    return data
E = take((V, D))
g1 = take((L, D))
Wq = take((L, D, D))
Wk = take((L, KV*HD, D))
Wv = take((L, KV*HD, D))
Wo = take((L, D, D))
g2 = take((L, D))
Wgate = take((L, F, D))
Wdown = take((L, D, F))
Wup = take((L, F, D))
gain = take((D,))
def norm(x, g):
    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(1e-5))) * g
def rotate(x):
    position = np.arange(x.shape[1], dtype=np.float32)
    frequency = np.power(np.float32(10000), -np.arange(0, HD, 2, dtype=np.float32)/HD)
    angle = position[:, None] * frequency[None, :]
    c, s = np.cos(angle), np.sin(angle)
    result = np.empty_like(x)
    result[..., 0::2] = x[..., 0::2]*c - x[..., 1::2]*s
    result[..., 1::2] = x[..., 0::2]*s + x[..., 1::2]*c
    return result
def forward(ids):
    x = E[ids].copy()
    observed = {}
    for l in range(L):
        y = norm(x, g1[l])
        q = rotate((y @ Wq[l].T).reshape(len(ids), H, HD).transpose(1,0,2))
        k = rotate((y @ Wk[l].T).reshape(len(ids), KV, HD).transpose(1,0,2))
        v = (y @ Wv[l].T).reshape(len(ids), KV, HD).transpose(1,0,2)
        heads = []
        for h in range(H):
            score = (q[h] @ k[h//(H//KV)].T) / np.sqrt(np.float32(HD))
            score[np.triu_indices(len(ids), 1)] = -np.inf
            exp = np.exp(score - np.max(score, axis=-1, keepdims=True))
            p = exp / np.sum(exp, axis=-1, keepdims=True)
            heads.append(p @ v[h//(H//KV)])
            if l == 0 and h == 0:
                observed['attention'] = p[-1].tolist()
                observed['query'] = q[0,-1].tolist()
        joined = np.stack(heads, axis=1).reshape(len(ids),D)
        x = x + joined @ Wo[l].T
        y = norm(x,g2[l])
        gate, up = y @ Wgate[l].T, y @ Wup[l].T
        x = x + ((gate / (1+np.exp(-gate))) * up) @ Wdown[l].T
    logits = norm(x,gain) @ E.T
    return logits[-1], observed

token_blob = (folder / 'tok512.bin').read_bytes()
assert hashlib.sha256(token_blob).hexdigest() == manifest['tokenizer']['sha256']
pieces, scores, cursor = [], [], 4
for i in range(V):
    score, length = struct.unpack_from('<fi',token_blob,cursor); cursor += 8
    pieces.append(token_blob[cursor:cursor+length].decode()); cursor += length
    scores.append(score)
lookup = {p:i for i,p in enumerate(pieces)}
def encode(text):
    ids = [1] + ([lookup[' ']] if text else [])
    for c in text:
        ids.extend([lookup[c]] if c in lookup else [b+3 for b in c.encode()])
    while True:
        candidates = [(scores[j],-i,j) for i in range(len(ids)-1) if (j:=lookup.get(pieces[ids[i]]+pieces[ids[i+1]])) is not None]
        if not candidates: return ids
        _, neg_i, j = max(candidates)
        ids[-neg_i:-neg_i+2] = [j]
def decode(ids):
    result = bytearray()
    for i,t in enumerate(ids):
        if t in (1,2): continue
        p = pieces[t]
        if i and ids[i-1] == 1 and p.startswith(' '): p = p[1:]
        result.extend(bytes([int(p[3:5],16)]) if p.startswith('<0x') else p.encode())
    return result.decode('utf-8', errors='replace')
prompt = 'Once upon a time'
ids = encode(prompt)
logits, observed = forward(ids)
output = []
for i in range(16):
    next_id = int(np.argmax(forward(ids+output)[0]))
    output.append(next_id)
    if next_id in (1,2): break
fixture = {'reference':'Independent NumPy FP32 full-prefix recomputation', 'checkpointSha256':manifest['weights']['sha256'], 'prompt':prompt, 'inputIds':ids, 'outputIds':output, 'text':decode(ids+output), 'lastQueryLogits':logits.tolist(), 'layer0Head0':observed, 'unicodeIds':encode('Hello, café 🧠!')}
destination = root / 'tests/fixtures/stories260k-reference.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(fixture,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in fixture.items() if k not in ('lastQueryLogits','layer0Head0')},indent=2))
