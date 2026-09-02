# What fits

Measured at load, not estimated -- see `src/ml_stack/data/fit.json`.

![How many fit, and what it costs](fit.png)

## This machine (110.0G)

### gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:52Z.

- weights 2.4G, compute 203.5M
- room 110.0G, of which 107.4G is left for caches
- **12.0K per token of context**, **12.0M fixed per sequence**
- 15 layers with a cache; a 1,024-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 1832 | 60.0M |
| 8,192 | 1017 | 108.0M |
| 16,384 | 538 | 204.0M |
| 32,768 | 277 | 396.0M |
| 65,536 | 140 | 780.0M |
| 131,072 | 71 | 1.5G |

One user, longest context: **9,380,311 tokens**.

### gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:45Z.

- weights 3.9G, draft 56.9M, compute 285.2M
- room 110.0G, of which 105.7G is left for caches
- **32.0K per token of context**, **40.0M fixed per sequence**
- 24 layers with a cache; a 1,024-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 644 | 168.0M |
| 8,192 | 365 | 296.0M |
| 16,384 | 196 | 552.0M |
| 32,768 | 101 | 1.0G |
| 65,536 | 51 | 2.0G |
| 131,072 | 26 | 4.0G |

One user, longest context: **3,463,598 tokens**.

### gpt-oss-120b-mxfp4-00001-of-00003.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:27:36Z.

- weights 59.0G, compute 181.1M
- room 110.0G, of which 50.8G is left for caches
- **72.0K per token of context**, **27.0M fixed per sequence**
- 36 layers with a cache; a 768-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 165 | 315.0M |
| 8,192 | 86 | 603.0M |
| 16,384 | 44 | 1.2G |
| 32,768 | 22 | 2.3G |
| 65,536 | 11 | 4.5G |
| 131,072 | 5 | 9.0G |

One user, longest context: **739,285 tokens**.

### Qwen3.8-27B-UD-Q4_K_XL.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:25Z.

- weights 16.7G, draft 1.3G, compute 302.5M
- room 110.0G, of which 91.7G is left for caches
- **128.0K per token of context**, **598.5M fixed per sequence**
- 16 layers with a cache; 64 recurrent (a fixed state per sequence, not per token)

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 84 | 1.1G |
| 8,192 | 57 | 1.6G |
| 16,384 | 35 | 2.6G |
| 32,768 | 20 | 4.6G |
| 65,536 | 10 | 8.6G |
| 131,072 | 5 | 16.6G |

One user, longest context: **746,718 tokens**.

### Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:24:41Z.

- weights 87.2G, compute 362.9M
- room 110.0G, of which 22.4G is left for caches
- **48.0K per token of context**, **256.6M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 51 | 448.6M |
| 8,192 | 35 | 640.6M |
| 16,384 | 22 | 1.0G |
| 32,768 | 12 | 1.8G |
| 65,536 | 6 | 3.3G |
| 131,072 | 3 | 6.3G |

One user, longest context: **483,795 tokens**.

### Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:29:39Z.

- weights 87.2G, draft 2.6G, compute 362.9M
- room 110.0G, of which 19.8G is left for caches
- **48.0K per token of context**, **594.3M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 25 | 786.3M |
| 8,192 | 20 | 978.3M |
| 16,384 | 14 | 1.3G |
| 32,768 | 9 | 2.1G |
| 65,536 | 5 | 3.6G |
| 131,072 | 3 | 6.6G |

One user, longest context: **419,897 tokens**.

### Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:25:18Z.

- weights 103.7G, compute 362.9M
- room 110.0G, of which 6.0G is left for caches
- **48.0K per token of context**, **256.6M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 13 | 448.6M |
| 8,192 | 9 | 640.6M |
| 16,384 | 5 | 1.0G |
| 32,768 | 3 | 1.8G |
| 65,536 | 1 | 3.3G |
| 131,072 | 0 | 6.3G |

One user, longest context: **124,663 tokens**.

## A machine with 24.0G

### gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:52Z.

- weights 2.4G, compute 203.5M
- room 24.0G, of which 21.4G is left for caches
- **12.0K per token of context**, **12.0M fixed per sequence**
- 15 layers with a cache; a 1,024-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 364 | 60.0M |
| 8,192 | 202 | 108.0M |
| 16,384 | 107 | 204.0M |
| 32,768 | 55 | 396.0M |
| 65,536 | 28 | 780.0M |
| 131,072 | 14 | 1.5G |

One user, longest context: **1,865,517 tokens**.

### gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:45Z.

- weights 3.9G, draft 56.9M, compute 285.2M
- room 24.0G, of which 19.7G is left for caches
- **32.0K per token of context**, **40.0M fixed per sequence**
- 24 layers with a cache; a 1,024-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 120 | 168.0M |
| 8,192 | 68 | 296.0M |
| 16,384 | 36 | 552.0M |
| 32,768 | 18 | 1.0G |
| 65,536 | 9 | 2.0G |
| 131,072 | 4 | 4.0G |

One user, longest context: **645,550 tokens**.

### gpt-oss-120b-mxfp4-00001-of-00003.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:27:36Z.

- weights 59.0G, compute 181.1M
- room 24.0G, of which 0B is left for caches
- **72.0K per token of context**, **27.0M fixed per sequence**
- 36 layers with a cache; a 768-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 0 | 315.0M |
| 8,192 | 0 | 603.0M |
| 16,384 | 0 | 1.2G |
| 32,768 | 0 | 2.3G |
| 65,536 | 0 | 4.5G |
| 131,072 | 0 | 9.0G |

One user, longest context: **0 tokens**.

### Qwen3.8-27B-UD-Q4_K_XL.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 3466812, 2026-09-02T18:26:25Z.

- weights 16.7G, draft 1.3G, compute 302.5M
- room 24.0G, of which 5.7G is left for caches
- **128.0K per token of context**, **598.5M fixed per sequence**
- 16 layers with a cache; 64 recurrent (a fixed state per sequence, not per token)

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 5 | 1.1G |
| 8,192 | 3 | 1.6G |
| 16,384 | 2 | 2.6G |
| 32,768 | 1 | 4.6G |
| 65,536 | 0 | 8.6G |
| 131,072 | 0 | 16.6G |

One user, longest context: **42,206 tokens**.

### Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:24:41Z.

- weights 87.2G, compute 362.9M
- room 24.0G, of which 0B is left for caches
- **48.0K per token of context**, **256.6M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 0 | 448.6M |
| 8,192 | 0 | 640.6M |
| 16,384 | 0 | 1.0G |
| 32,768 | 0 | 1.8G |
| 65,536 | 0 | 3.3G |
| 131,072 | 0 | 6.3G |

One user, longest context: **0 tokens**.

### Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

f16 cache, guessing ahead by draft-mtp, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:29:39Z.

- weights 87.2G, draft 2.6G, compute 362.9M
- room 24.0G, of which 0B is left for caches
- **48.0K per token of context**, **594.3M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 0 | 786.3M |
| 8,192 | 0 | 978.3M |
| 16,384 | 0 | 1.3G |
| 32,768 | 0 | 2.1G |
| 65,536 | 0 | 3.6G |
| 131,072 | 0 | 6.6G |

One user, longest context: **0 tokens**.

### Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

f16 cache, measured at 32,768 tokens over 2 slots, build 92cedc867, 2026-09-02T18:25:18Z.

- weights 103.7G, compute 362.9M
- room 24.0G, of which 0B is left for caches
- **48.0K per token of context**, **256.6M fixed per sequence**
- 24 layers with a cache; 48 recurrent (a fixed state per sequence, not per token); a 16,384-cell sliding window per sequence

| per user context | users that fit | each costs |
| --- | --- | --- |
| 4,096 | 0 | 448.6M |
| 8,192 | 0 | 640.6M |
| 16,384 | 0 | 1.0G |
| 32,768 | 0 | 1.8G |
| 65,536 | 0 | 3.3G |
| 131,072 | 0 | 6.3G |

One user, longest context: **0 tokens**.
