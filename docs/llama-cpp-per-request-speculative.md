# Per-request draft depth for `llama-server`

A patch for `ggml-org/llama.cpp`, carried in `patches/llama.cpp/` and applied by
`ml-stack-serve build --from source`. Not sent upstream; this note is what a pull request
would say.

## What it enables

`llama-server` binds draft depth when the server starts (`--spec-draft-n-max`). One served
model therefore has one depth, so measuring seven depths costs seven loads, and two
workloads wanting different depths cannot share a server.

With the patch a request carries its own:

```json
{"messages": [...], "speculative.n_max": 4}
```

A request may draft *less* than the server started with, never more: the draft
implementations size their buffers from the startup value. A sweep therefore serves once at
the deepest depth it means to ask for and asks each depth of that one load.

Measured on gemma-4-E2B-it-qat (Q4_K_XL) with its MTP head, served at
`--spec-draft-n-max 8`, 96 tokens generated per request, greedy:

| request | drafted | accepted | tok/s |
| --- | --- | --- | --- |
| no field | 150 | 76 | 141.3 |
| `speculative.n_max: 1` | 52 | 42 | 169.6 |
| `speculative.n_max: 2` | 76 | 56 | 176.3 |
| `speculative.n_max: 4` | 101 | 69 | 176.6 |
| `speculative.n_max: 8` | 150 | 76 | 141.3 |

Depth 8 reproduces the no-field row, which is what asking for the depth the server started
at should do.

## Why the guard was there

The registrations exist in `tools/server/server-schema.cpp` behind:

```c
// TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
```

The block does not compile as written, which is consistent with its never having been built
since the schema was introduced:

- The `speculative.n_min` registration ends `->set_desc("...");` — the closing paren for
  `add(` is missing.
- `speculative.ngram_size_n`, `ngram_size_m` and `ngram_min_hits` name `uint16_t` fields, so
  they instantiate `field_num<unsigned short>`, whose `eval` calls
  `common_json::get<unsigned short>()`. `common/json.cpp` instantiates `get<T>` for an
  explicit list of types that does not include `unsigned short`, so the build fails at link
  with an undefined symbol rather than at compile.

## What the patch registers, and what it leaves alone

Only `speculative.n_max` is registered. The other six fields in the block are read once, by
the speculative implementations, when they are constructed at server start:
`common_speculative_impl_draft_*` copies `common_params_speculative` into itself, and
nothing updates that copy per request. Registering them would accept a value and change
nothing, so they stay guarded with a line saying why.

This was measured before it was decided. With every field registered, against the same
served model:

| request | drafted | accepted |
| --- | --- | --- |
| no field | 150 | 76 |
| `speculative.p_min: 0.99` | 150 | 76 |
| `speculative.n_min: 8` | 150 | 76 |

Leaving the ngram fields unregistered also means the `unsigned short` instantiation is not
needed, so `common/json.cpp` is untouched.

## The three files

`tools/server/server-schema.cpp` — `speculative.n_max` moved out of the guard, the missing
paren fixed so the remaining block is not left broken, and the TODOs replaced by a line
naming what the guarded fields depend on.

`tools/server/server-context.cpp` — `server_slot::get_n_draft_max()` clamped by
`task->params.speculative.draft.n_max`. Without this the schema writes the request's value
into the task and nothing reads it: the slot's draft budget was computed from the context
and the tokens remaining only, so the field would be accepted and ignored. The clamp is what
carries it into `common_speculative_draft_params::n_max`, the per-sequence override the
drafting code already has.

`common/speculative.cpp` — the eagle3 and MTP draft loops stop at that per-sequence cap the
way the simple draft loop already does:

```c
if ((params.n_max <= (int) result.size()) ||
    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
```

`common_speculative_gen_draft` already truncates every implementation's result to
`dp.n_max`, so the depth was correct without this. What it was not was cheaper: a request
asking for depth 2 against a server started at 8 drafted eight tokens and threw six away,
which makes a depth sweep read acceptance correctly and latency wrongly. The dflash and
dspark implementations size one block up front and still draft the startup depth; they are
correct, not yet cheaper.

## What a server without the patch does

It ignores the field in silence. The schema evaluates the fields it knows and never looks at
the rest of the body, so an unknown key is neither read nor refused. Measured against the
same model on the stock mainline build, every depth returned the same 150 drafted and 76
accepted, and `speculative.n_max: -1` — outside the registered field's hard limits — was
answered `200 OK` rather than the `400` the patched build returns.

A caller therefore cannot tell from a successful response whether the depth took.
`ml_stack.bench.backends.draft_depth_support` measures it instead, with one call carrying no
field and one asking for depth 0, and reads `obeyed`, `ignored` or `no drafting`.

## The field is flat, not an object

The schema registers the literal key `"speculative.n_max"`, and the driver asks each
registered field whether the body contains its own name. A body carrying
`{"speculative": {"n_max": 1}}` therefore matches nothing. Measured against the patched
build, started at depth 8: the flat form drafted 52 tokens and accepted 42, while the
nested form drafted 150 and accepted 76 -- the same as sending no field at all.
