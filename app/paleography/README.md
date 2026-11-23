# paleography — M4M training module

Trains **M4M** (morphology4metrology, the architecture and training recipe of [3]) on a dataset of
text-line images with their transcriptions (the D1 dataset format), and returns, for the dataset:
character prototypes, per-line character measurements and the character error rate (CER).

The trainer is the git submodule `lib/m4m` (fork `NassimaOULDOUALI/morphology4metrology`,
branch `aikon-api`). What that branch changes with respect to the upstream repository is
documented in `AIKON_API.md` at the root of the fork.

## Routes (prefix `/paleography`)

| Method | Route | Purpose |
|---|---|---|
| POST | `start` | Start a task (shared `receive_task` / `start_task`) |
| GET | `<tracking_id>/status` | Task state, including epoch progress (see below) |
| POST | `<tracking_id>/cancel` | Cancel a task |
| GET | `<doc_id>/result` | Results zip (`data/paleography/results/<doc_id>.zip`) |
| GET | `qsizes`, `monitor` | Shared routes (queue sizes, monitoring) |

The task runs on the Dramatiq queue `queue5`.

## Input

One document of type `zip` per dataset, organised as:

```
<folio>/<line>.png   (or .jpg / .jpeg)
<folio>/<line>.txt   (transcription, same file name)
```

- Folder names are used as sub-dataset (folio) names. No script information is required.
- `.tif` / `.tiff` images are rejected by the converter.
- A dataset with more than one document, or a document that is not a zip, is refused.
- `.txt` files are kept at extraction because `.txt` is in `ALLOWED_EXTENSIONS`
  (`app/shared/dataset/document.py`).

## Request

```bash
curl -X POST localhost:5001/paleography/start -H 'Content-Type: application/json' -d '{
  "experiment_id": "my_run",
  "notify_url": "<front notification URL>",
  "documents": [{"uid": "my_run", "type": "zip", "src": "<URL of the D1 zip>"}],
  "parameters": {"cuda_device": "0",
                 "step0": {"max_e": 2}, "step1": {"max_e": 2}, "step2": {"max_e": 2}}
}'
```

No local path is needed: the converter and the step-0 checkpoint have defaults.

### Parameters (all optional)

| Parameter | Default | Notes |
|---|---|---|
| `cuda_device` | unset (all visible GPUs) | Sets `CUDA_VISIBLE_DEVICES` for every M4M subprocess, export included |
| `seed` | `42` | Passed as `--seed`; `null` disables it |
| `step0` | `max_e` 20, `batch_size` 8 | Step 0 adapts the checkpoint heads to the dataset charset (`--init`) |
| `step1` | `max_e` 100, `batch_size` 8, `learning_rate` 1e-4, `weight_loss_reconstruction` 3 | Full training |
| `step2` | `max_e` 140, `batch_size` 8, `learning_rate` 1e-2 | Fine-tuning, one run per folio (`--documents`) |
| `checkpoint` | `data/paleography/models/checkpoint.pth` | Downloaded from Hugging Face (`RaphaelBfr/morphology4metrology`) when `const.py` is imported |
| `converter_path` | `lib/convert_d1_to_d2.py` | D1 → M4M dataset converter |
| `script`, `script_map` | none | Optional converter options |
| `charset`, `annotation_file` | none | Optional export options |
| `steps` | all: `convert, 0, 1, 2, export` | Internal: run a subset, always in this order |
| `data_folder`, `run_id` | `paleo_<experiment_id>` | Internal: with `steps`, resume from an earlier run |

`max_e N` runs exactly N epochs (fixed in the fork).

On an 11 GB GPU (RTX 2080 Ti), step 2 ran out of GPU memory with `batch_size` 8 on some folios
of a 4,438-line dataset; it completed with `"step2": {"batch_size": 4}`.

Resuming only step 2 and the export from an earlier run:

```json
"parameters": {"cuda_device": "0", "steps": ["2", "export"],
               "data_folder": "paleo_<earlier experiment_id>", "run_id": "paleo_<earlier experiment_id>",
               "step2": {"max_e": 5, "batch_size": 4}}
```

## Pipeline

`convert` → step 0 → step 1 → step 2 → `export`, in one Dramatiq actor (`tasks.run_paleography`).
Each M4M step runs as a subprocess with the API interpreter (`sys.executable`), from `lib/m4m`.

| Path | Content |
|---|---|
| `data/paleography/datasets/<data_folder>` | Converted dataset (M4M format, `annotation.json`) |
| `data/paleography/logs/<run_id>_s{0,1,2}` | M4M outputs (`lib/m4m/logs_reconstruction` is a link to `data/paleography/logs`) |
| `data/paleography/models/checkpoint.pth` | Step-0 checkpoint |
| `data/paleography/results/<doc_uid>.zip` | Results served by `<doc_id>/result` |

`lib/m4m/datasets/config.json` is written on the first run if absent (`datasets_path` →
`data/paleography/datasets`).

## Output

The SUCCESS notification contains `results_url = [{"doc_id": ..., "result_url": ...}]`.
The zip contains:

```
characters_measurements/<folio>/...json   one JSON per line
prototypes/baseline_with_ar/              sprites
prototypes/baseline_without_ar/           sprites
prototypes/<folio>/                       sprites, one folder per folio
cer/global_summary.json                   CER over all lines
cer/<folio>/document_summary.json         CER per folio
```

The task fails if the export produced no prototype folder, an empty prototype folder, no
measurement JSON, or no `global_summary.json`.

The CER is computed on the lines used for training (no train/validation split): it shows how
well the model fits these lines, not how it generalises.

## Progress

While the task runs, `<tracking_id>/status` returns a single progress entry in `log.progress`:

```json
{"context": "Step 2 - btv1b84472995_f015", "current": 2, "total": 5}
```

`context` is `Step 0`, `Step 1`, `Step 2 - <folio>` or `Export (predict + measurements)`;
`current` is the number of completed epochs. It is read from M4M's own `Epoch=N/T` output,
which is still written to the worker log.

## Installation (development)

Tested on a fresh clone on 23/11/2026 (Ouranos, 2× RTX 2080 Ti, CUDA 12.6).

```bash
git submodule update --init app/paleography/lib/m4m
uv sync --python 3.11                     # torch, editdistance, wandb from the lock

# CUDA ops (MultiScaleDeformableAttention): not in the lock, removed by `uv sync`
A=$PWD
uv pip install --python .venv/bin/python setuptools
cp -a app/region_extraction/lib/line_predictor/dino/ops /tmp/ops_build
( cd /tmp/ops_build && CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
  TORCH_CUDA_ARCH_LIST="7.5" $A/.venv/bin/python setup.py build install )
.venv/bin/python -c "import torch, MultiScaleDeformableAttention; print('ops OK')"

# worker on the paleography queue
.venv/bin/python -m dramatiq app.main -p 1 -t 1 -Q queue5
```

- Target the `.venv` explicitly in `uv pip` commands: with a non-base conda environment active,
  `uv pip install` installs into the conda environment.
- `TORCH_CUDA_ARCH_LIST="7.5"` matches the RTX 2080 Ti; adapt it to the GPU.
- An expired Hugging Face token stored on the machine (`~/.cache/huggingface/token`) makes the
  checkpoint download fail with a 401, although the repository is public.

In Docker, `docker/start.sh` builds the ops when `paleography` is in `INSTALLED_APPS`
(not tested with the image).

## Known limitations

- M4M runs as a subprocess, not as a direct call (`reconstruction.py` parses its arguments at
  import time).
- With the same seed, two runs do not produce identical files (CER 0.471 to 0.478 on the test
  set); likely cause: non-deterministic CUDA ops.
- M4M outputs are kept after the zip is created (about 8 GB for 4,438 lines, 5 epochs per step).
- A task redelivered after a worker stop fails: the converter refuses an existing output folder.
- `queue5` only has a worker when `NB_PROCS` ≥ 6 (`install.py` sets `NB_PROCS` to the number of
  installed apps).
- The checkpoint is downloaded when `const.py` is imported: if the download fails, the API and
  the worker do not start.
- With several documents in a dataset, the same zip is copied for each document.
