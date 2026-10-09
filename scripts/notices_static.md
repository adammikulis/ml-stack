## Optional extras with copyleft licences

These are never bundled and never installed by default. They are imported only when you install them yourself.

| Extra | Package | Licence | Effect |
|---|---|---|---|
| `speech` | piper-tts | GPL-3.0-or-later | `ml_stack.speech.tts` imports `piper` at run time. ml-stack does not distribute piper. Distributing a build of ml-stack together with piper would have to satisfy the GPL for the combination. |
| `pdf-agpl` (in neither `all` nor `redteam`) | pymupdf (MuPDF) | AGPL-3.0 or commercial | Opt-in engine only: `ML_STACK_PDF_ENGINE=pymupdf` makes `ml_stack.sources.pdf` and `ml_stack.net.pdftext` import `pymupdf`, and the datasheet pin tables need it (page pictures and outline pages do not). The default `pdf` extra reads with pdfminer.six (MIT) and Pillow and renders pages with PDFium through pypdfium2 (BSD-3-Clause / Apache-2.0). Offering a service built on pymupdf over a network triggers the AGPL for that service. |

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
