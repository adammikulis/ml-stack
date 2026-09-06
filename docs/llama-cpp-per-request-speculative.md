# Per-request speculative decoding parameters for `llama-server`

A patch for `ggml-org/llama.cpp`, carried in `patches/llama.cpp/` and applied by
`ml-stack-serve build --from source`. Not sent upstream; this note is what a pull request
would say.

## What it enables

`llama-server` binds draft depth when the server starts (`--spec-draft-n-max`). One
served model therefore has one depth, so measuring seven depths costs seven loads, and two
workloads that want different depths cannot share a server.

With the patch a request carries its own:

```json
{"messages": [...], "speculative.n_max": 4}
```

A request may draft *less* than the server started with, never more: the buffers the draft
implementations allocate are sized from the startup value. So a sweep serves once at the
deepest depth it means to ask for and asks each depth of that one load.

## Why the guard was there

The registrations exist in `tools/server/server-schema.cpp`, behind:

```c
// TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
```

The block does not compile as written, which is consistent with its never having been
built since the schema was introduced:

- The `speculative.n_min` registration ends `->set_desc("...");` — the closing paren for
  `add(` is missing.
- `speculative.ngram_size_n`, `ngram_size_m` and `ngram_min_hits` name `uint16_t` fields,
  so they instantiate `field_num<unsigned short>`, whose `eval` calls
  `common_json::get<unsigned short>()`. `common/json.cpp` instantiates `get<T>` for an
  explicit list of types and `unsigned short` is not on it, so the build fails at link
  with an undefined symbol rather than at compile.

## What the patch changes

`common/json.cpp` — one line, `COMMON_JSON_GET(unsigned short)`, added to the existing
list.

`tools/server/server-schema.cpp` — the missing paren, and the `#if 0` and its two TODOs
removed.

`tools/server/server-context.cpp` — `server_slot::get_n_draft_max()` clamped by
`task->params.speculative.draft.n_max`. Without this the schema writes the request's value
into the task and nothing reads it: the slot's draft budget was computed from the context
and the tokens remaining only, so the field would be accepted and ignored. The clamp is
what carries it into `common_speculative_draft_params::n_max`, the per-sequence override
the drafting code already has.

`common/speculative.cpp` — the eagle3 and MTP draft loops stop at the per-sequence cap the
way the simple draft loop already does:

```c
if ((params.n_max <= (int) result.size()) ||
    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
```

`common_speculative_gen_draft` already truncates every implementation's result to
`dp.n_max`, so the depth was correct without this. What it was not was cheaper: a request
asking for depth 2 against a server started at 8 drafted eight tokens and threw six away,
which makes a depth sweep measure acceptance correctly and latency wrongly. The dflash and
dspark implementations size one block up front and still draft the startup depth; they are
correct, not yet cheaper.

## What it does not do

- A request cannot raise the depth above the server's. Raising it means reallocating
  buffers sized at construction.
- `speculative.type` switches the method per request. It is registered as upstream wrote
  it and is not exercised here.
