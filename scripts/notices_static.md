## Optional extras with copyleft licences

These are never bundled and never installed by default. They are imported only when you install them yourself.

| Extra | Package | Licence | Effect |
|---|---|---|---|
| `speech` | piper-tts | GPL-3.0-or-later | `ml_stack.speech.tts` imports `piper` at run time. ml-stack does not distribute piper. Distributing a build of ml-stack together with piper would have to satisfy the GPL for the combination. |
| `pdf` (also in `all`) | pymupdf (MuPDF) | AGPL-3.0 or commercial | `ml_stack.sources.pdf` imports `pymupdf` at run time. Offering a service built on it over a network triggers the AGPL for that service. |

Other optional extras (`torch`, `transformers`, `peft`, `mlx*`, `manim`, `spacy`, `presidio-analyzer`, `playwright`,
`lm-eval` and others) are permissively licensed at the top level, but their own dependencies are not inventoried
here; run `pip-licenses` in the environment you build from before redistributing it.

## Code ported into this repository

See `NOTICE` (qwen-spec, DFlash, mlx-lm, SGLang).

## Programs run as separate processes

llama.cpp (`llama-server`, MIT), ffmpeg, and the models you download are separate programs with their own
licences; none is bundled in the wheel or in the release bundles. The bundles are built with PyInstaller
(GPL-2.0 with the bootloader exception, which permits distributing the frozen application under any licence)
and Tauri (MIT or Apache-2.0).
