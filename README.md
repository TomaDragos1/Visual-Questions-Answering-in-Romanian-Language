# Romanian Visual Question Answering

Fine-tuning and evaluating vision-language models (Qwen2-VL, Gemma 3, Llama 3.2 Vision)
for Visual Question Answering in **Romanian**, with two retrieval-augmented variants:
a vLLM zero-shot baseline and a RAG-based few-shot pipeline.

The training data is the [GQA](https://cs.stanford.edu/people/dorarad/gqa/) dataset
retranslated to Romanian; evaluation is done on Romanian GQA and a Romanian subset of
VQAv2.

---

## LoRA adapters on Hugging Face

The fine-tuned LoRA adapters trained with the notebook `01_finetune.ipynb` are
published on Hugging Face. You can load them directly without retraining:

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name   = "<your-hf-username>/<adapter-name>",
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)
```

There are 2 repos for models that i fine-tunned: 					
Toma1/qwen3vl-vqa-romana-lora
Toma1/gemma4-vqa-romana-lora

You can play with this models and RAG

Anyone can use these adapters — just plug the HF repo id into the `MODEL_PATH`
variable in `02_inference_finetuned.ipynb` (or `04` / `05`) and run.

---

## Notebooks

All notebooks are designed to run on **Google Colab** with an A100 (40GB or 80GB).
Smaller GPUs work for the inference notebooks if you drop the model size.

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `01_finetune.ipynb` | Fine-tunes a 4-bit vision-language model on Romanian GQA with Unsloth + LoRA. Saves the adapter to Drive. |
| 02 | `02_inference_finetuned.ipynb` | Loads the fine-tuned adapter, runs inference on the Romanian GQA test set, and evaluates with accuracy / F1 / BLEU / ROUGE / BERTScore. |
| 03 | `03_retranslate_and_build_rag.ipynb` | Retranslates the full 900k English VQA pairs to Romanian with vLLM (continuous batching). Then builds a ChromaDB RAG index over the retranslated questions (multilingual MiniLM) and unique images (CLIP). |
| 04 | `04_rag_fewshot_inference.ipynb` | Few-shot inference: retrieves the top-k similar (question, image) demos from the RAG index using a hybrid text + image score, builds a multimodal prompt, and runs the model. Supports both GQA and VQAv2 test sets. |
| 05 | `05_vllm_zeroshot_inference.ipynb` | Zero-shot baseline using vLLM with large batches. Includes an optional step to export an Unsloth fine-tuned LoRA into a merged folder that vLLM can load. |

---

## How to run

### 1. Prerequisites

- A Google Colab account with access to an A100 GPU (Colab Pro+ or a cloud A100)
- A Google Drive with:
  - The GQA images, zipped (`images.zip`)
  - The Romanian-translated JSON files (train / val / test)
  - For VQAv2 evaluation: the Romanian VQAv2 subset and its COCO images zip
- A Hugging Face account (for downloading base models and optionally pushing your own adapter)

### 2. Pipeline order

The notebooks are independent, but the natural order is:

1. **`03_retranslate_and_build_rag.ipynb`** — only needed if you don't already have
   the retranslated dataset and the RAG index. Skip it if you only want to fine-tune
   or run zero-shot.
2. **`01_finetune.ipynb`** — fine-tunes the model. Set `BASE_MODEL` and `SAVE_DIR` at
   the top of the notebook.
3. **`02_inference_finetuned.ipynb`** — evaluates the fine-tuned model on the test set.
4. **`04_rag_fewshot_inference.ipynb`** — runs RAG-based few-shot inference (requires
   the index from notebook 03).
5. **`05_vllm_zeroshot_inference.ipynb`** — runs the zero-shot baseline with vLLM
   for comparison.

### 3. Running a notebook

- Open it in Colab, set the runtime to GPU (A100)
- Mount Drive when prompted
- Edit the **Config** cell at the top to point at your own paths and chosen model
- Run all cells

All inference notebooks are **resume-safe**: predictions are streamed to a JSONL
and already-processed `questionId`s are skipped on the next run.

---

## Just use the adapter (no fine-tuning)

If you only want to try the fine-tuned model:

1. Open `02_inference_finetuned.ipynb`
2. Set `MODEL_PATH = "<your-hf-username>/<adapter-name>"`
3. Set `TEST_JSON` to your Romanian VQA test file (or build a small one)
4. Run all cells

---

## Project structure

```
.
├── README.md
└── notebooks/
    ├── 01_finetune.ipynb
    ├── 02_inference_finetuned.ipynb
    ├── 03_retranslate_and_build_rag.ipynb
    ├── 04_rag_fewshot_inference.ipynb
    └── 05_vllm_zeroshot_inference.ipynb
```

---

## Stack

- **Unsloth** — fast 4-bit LoRA training and inference for vision-language models
- **vLLM** — high-throughput batched inference and dataset retranslation
- **ChromaDB** — persistent vector store for the RAG index
- **Sentence-Transformers** — multilingual MiniLM for question embeddings, CLIP for image embeddings
- **TRL** — `SFTTrainer` for supervised fine-tuning
- **HuggingFace `transformers`, `datasets`, `peft`**

## Models tested as backbones

- `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit`
- `unsloth/qwen3-vl-32b-instruct-unsloth-bnb-4bit`
- `unsloth/gemma-4-31b-it-unsloth-bnb-4bit`
- `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`

## Datasets

- **Training:** Romanian-translated GQA (retranslated with Gemma / Qwen via vLLM)
- **Evaluation:** Romanian GQA test set, Romanian VQAv2 subset (15k)

---

## Notes

- Notebook prompts (`INSTRUCTION_PROMPTS`, `INSTRUCTION`) are kept in Romanian on
  purpose — that is the target language for the model. Code comments are in English.
- The RAG index is rebuilt locally on first run (copied off Drive) for stability.
- All inference loops checkpoint to JSONL — disconnecting from Colab and resuming is safe.
