# Medical decision-model fine-tuning

## Scope

This run adapts the typed-choice decider to medical exam questions from MedMCQA. It is a research
benchmark experiment, not a clinical triage or treatment system. The cases contain exam questions,
not patient records. Do not use the resulting model to make care decisions.

## Pinned inputs

| Artifact | Repository | Revision | Local status |
|---|---|---|---|
| Training and evaluation cases | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa/tree/91c6572c454088bf71b679ad90aa8dffcd0d5868) | `91c6572c454088bf71b679ad90aa8dffcd0d5868` | Cached; Apache-2.0 dataset card |
| Base model | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base/tree/b1485b2fa6dfa1287294f269f5fb618e03d52d7c) | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | Cached; pinned by `poolhouse.decide.pins` |
| Decision checkpoint | [StrandsAgents/strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19/tree/bb282d786bc251fd4e3068de3ada9ddbb38127cd) | `bb282d786bc251fd4e3068de3ada9ddbb38127cd` | Download requested for this run |
| Clef-flash comparison | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6) | `17f0b0ad64efb65d273590632833508766b2aae6` | Cached; not supported by this training recipe |
| Laya comparison | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/e929ae5cf69bc34259cd2f95c9e91145b818b1f0) | `e929ae5cf69bc34259cd2f95c9e91145b818b1f0` | Cached; not supported by this training recipe |

The MedMCQA card describes 182,822 training rows and a 4,183-row validation split. This
preparation keeps labelled single-answer rows, uses the four answer choices as options A–D, and
uses the validation split only for evaluation. Explanations and the unlabeled test split are not
included. The original dataset card describes this as medical entrance-exam question answering;
it is not a patient-triage dataset.

## Reproduce the local inputs

The model and dataset files stay in the Hugging Face cache. Fine-tuned weights and generated case
files stay in the Poolhouse state/cache directories and are not committed.

```bash
poolhouse-models snapshot openlifescienceai/medmcqa --repo-type dataset --revision 91c6572c454088bf71b679ad90aa8dffcd0d5868
poolhouse-models snapshot Qwen/Qwen3.5-2B-Base --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
poolhouse-models snapshot StrandsAgents/strands-decider-2B-hobson-v19 --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd
poolhouse-models snapshot Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6
poolhouse-models snapshot convaiinnovations/laya-typed-decisions --revision e929ae5cf69bc34259cd2f95c9e91145b818b1f0
```

Install the `decide-data` extra in the data-preparation environment, then convert the cached dataset
snapshot. The generated files must remain outside the Git checkout. Windows-prepared files are
available in the owner's cache under `poolhouse/medical/medmcqa-91c6572c`; WSL can read that cache
through its mounted Windows filesystem.

```bash
snapshot="$HOME/.cache/huggingface/hub/datasets--openlifescienceai--medmcqa/snapshots/91c6572c454088bf71b679ad90aa8dffcd0d5868"
cases="$HOME/.cache/poolhouse/medical/medmcqa-91c6572c"
python scripts/prepare_medmcqa.py "$snapshot" "$cases"
```

The Windows preparation on 2026-10-05 using `scripts/prepare_medmcqa.py` against the pinned
snapshot produced 120,765 training cases (SHA-256
`2d9c71b2acc6735afe79d97a5ebf7d763a8c5ff51839161359f5512d6d687976`) and 2,816 validation
cases (SHA-256 `ef86aab609aa02a8a42362ccb45743c0fd8a847b2eb6c21ba86aec2f42a07c18`). Recheck
these hashes when reusing the files from WSL.

## First run

Start with 120 steps from the Strands pointer checkpoint and compare against that same checkpoint.
The trainer checks a group-held-out split within the training file, fits calibration on its own
held-out groups, and scores the official validation file. It registers the result only if it does
not lose to the baseline. The GPU broker grants the training lease.

```bash
poolhouse-decide train --data "$cases/train.jsonl" --eval "$cases/validation.jsonl" \
  --name medical-medmcqa-strands-v19 --base qwen3.5-2b-base --init strands \
  --baseline strands --steps 120 --batch-size 4 --lr 5e-5 --device cuda --wait 3600
```

Record the date, installed wheel version, source revision, data hashes, split counts, exact
command, GPU, step losses, baseline result, final accuracy, Brier score, calibration error,
abstention rate and output directory in a run record. Keep the model card and weights in the
local model cache. A benchmark pass does not establish clinical safety or support deployment.

## Downloaded comparison models

The cached Clef-flash and Laya checkpoints are exploratory comparisons. The Poolhouse pointer
trainer accepts the pinned Qwen3.5 base and Strands pointer checkpoint; it does not fine-tune
these two alternative architectures. Their revisions above let another device fetch the same
files without putting model weights in Git.
