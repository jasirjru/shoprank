# ShopRank lessons: section 15, serving the model (built 21 Nov 2026, study after the build)

Served model: MiniLM distilled from DeBERTa, reading title + brand + colour + bullets
(`minilm_distil_w10_bullets`; test nDCG 0.8674). Numbers measured on this laptop (16 cores / 32 threads).
Answer the check questions first; the key is at the bottom.

---

## Part 1: ONNX and INT8: making the model small and fast on a CPU

**Picture:** PyTorch is a full kitchen where you can invent recipes. ONNX is the printed recipe card that any
restaurant (ONNX Runtime, on any machine) can cook from, without the kitchen. INT8 is writing the amounts in
whole spoons instead of milligrams: a little less exact, much quicker to read.

**Real numbers (dev, 40,304 pairs):**

| Version | File size | dev nDCG | Max score difference vs PyTorch fp32 |
|---|---|---|---|
| PyTorch (training eval, bf16) | 127 MB | 0.8667 | |
| ONNX fp32, fused kernels | 127 MB | 0.8662 | 0.00000095 |
| **ONNX INT8** | **32 MB** | **0.8654** | 0.087 |

```
(illustrative)
weights fp32:  0.01734  -0.20918   0.11502      4 bytes each
weights INT8:      1        -12         7       1 byte each, x scale 0.0173 (one scale per output channel)
```

| Word | Meaning |
|---|---|
| ONNX | a standard file format for a trained network, independent of PyTorch |
| fusion | merging several small steps (e.g. attention parts) into one fast kernel |
| dynamic INT8 | weights stored as 8-bit integers; activations converted per batch while running |
| parity check | proof the exported model computes the same function as the original |

**Interview answer:** "I exported the distilled MiniLM to ONNX, fused the attention and LayerNorm kernels with
ONNX Runtime's transformer optimizer, and applied dynamic INT8 quantisation per channel. The fp32 export
matched PyTorch to 1e-6; INT8 made the model 4x smaller and cost 0.0008 nDCG on dev."

## Part 2: The API: validate at the door

**Picture:** a bank counter checks your form before it reaches the cashier. A wrong form is sent back with the
reason; the cashier's time is never wasted on it.

`POST /rank` with a query and its candidate products returns them best first:
```
{"query": "usb c cable 6ft",
 "products": [{"product_id": "B01", "title": "Anker USB C cable", "brand": "Anker", "bullets": "6 ft ..."}, ...]}
-> {"model": "...", "results": [{"product_id": "B01", "score": 0.93}, ...], "latency_ms": 64.2}
```
(Since v1.1 the response also has `"query_used"`: the query after spell correction, see section 18.)
- Rejected with 422 before any model call: blank or > 200-character query, 0 or > 200 products, missing title,
  duplicate ids, unknown fields, over-long text.
- A model failure returns 500 "scoring failed"; the details go to the server log only.
- The product text is built exactly as in training (a test checks it): a different join at serving time would
  silently change the input ("training/serving skew").

**Interview answer:** "The API validates every request with Pydantic at the boundary, so bad input never
reaches the model, hides internal errors from clients, and has a test proving the serving text equals the
training text, which is the most common silent bug in served rankers."

## Part 3: Load testing: p50, p95, p99, and why more parallel requests made it slower

**Picture:** one cook with 16 hands. If 16 customers each grab all 16 hands at once, the hands get tangled and
fewer meals come out. Better: customers queue, the cook uses all hands on one order at a time.

**Real numbers (requests = real dev queries, median 16 products each):**

| Setting | Parallel requests | p50 | p95 | pairs/s |
|---|---|---|---|---|
| every call uses all 32 threads | 1 | 68 ms | 233 ms | 222 |
| every call uses all 32 threads | 16 | 1,880 ms | 4,018 ms | **151** (fell) |
| one call at a time, 16 threads | 1 | 69 ms | 232 ms | 217 |
| one call at a time, 16 threads | 16 | 1,436 ms | **1,784 ms** | **230** (held) |

- p50 = half of requests are faster than this; p95 = 95% are; p99 = the slow tail users still notice.
- At 16 parallel requests the latency is queueing: ~11 requests/s is the honest capacity of one laptop
  process. More capacity = more machines (or processes on more cores), not more threads.

**Quality vs speed:**

| Option | Test nDCG | Speed | Needs |
|---|---|---|---|
| BM25 | 0.8202 | very fast | CPU |
| **MiniLM distilled, INT8 (served)** | 0.8674 (PyTorch; INT8 -0.0008 on dev) | 230 pairs/s | CPU |
| DeBERTa x5 | 0.8877 | ~89 pairs/s | GPU |

**Interview answer:** "I load-tested the API with real queries. With every request using all cores,
throughput fell 32% under 16 parallel requests because threads fought over the CPU. Limiting the process to
one model call at a time on the 16 physical cores kept throughput flat at 230 pairs per second and cut p95
from 4.0 to 1.8 seconds at that load. The distilled INT8 model serves at 69 ms p50 on a CPU; the 0.02-better
DeBERTa ensemble would need a GPU and is still 2.6x slower, so it stays the offline teacher."

---

## Check questions

1. (Part 1) What did the parity check (max difference 0.00000095) prove? A) INT8 is exact B) the ONNX fp32
   graph computes the same function as PyTorch C) the model is faster D) dev nDCG went up
2. (Part 1) Why is the INT8 file 4x smaller? A) fewer layers B) 1 byte per weight instead of 4 C) the
   vocabulary was cut D) bullets were removed
3. (Part 2) Why test that serving text equals training text? A) speed B) a different input format at serving
   time silently lowers quality (training/serving skew) C) security D) to save memory
4. (Part 3) Why did throughput FALL from 222 to 151 pairs/s with 16 parallel requests? A) the network was slow
   B) 16 calls x 32 threads fought over 16 cores C) INT8 broke D) the GPU overheated
5. (Part 3) p95 = 1,784 ms means: A) the average is 1.8 s B) 95% of requests finished within 1.8 s
   C) 5% finished within 1.8 s D) the fastest took 1.8 s

## Answer key

1. B. 2. B. 3. B. 4. B. 5. B.
