# MBG Sentiment Analysis

Live demo: https://nzlul03.github.io/mbg-sentiment-analysis/

A one-page app that classifies Indonesian tweets about the Free Nutritious Meals program (Makan Bergizi Gratis (MBG)) as **Negative**, **Neutral** or **Positive**. Inference runs entirely in the browser with [Transformers.js](https://github.com/huggingface/transformers.js); no server is involved.

| Option | Model | How it runs |
|---|---|---|
| IndoBERT | [`Nzlul/indobert-mbg-sentiment`](https://huggingface.co/Nzlul/indobert-mbg-sentiment): [IndoBERT-lite](https://huggingface.co/indobenchmark/indobert-lite-base-p1) fine-tuned on 770 labelled tweets, ONNX (~47 MB) | WebAssembly (CPU) |
| Qwen 3.5 | [`onnx-community/Qwen3.5-0.8B-ONNX-OPT`](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT), 4-bit (~600 MB) | WebGPU, few-shot prompt, label scoring |

Test-set results from the project notebook (165 tweets): IndoBERT-lite F1-macro 0.792, Qwen3.5-0.8B zero-shot F1-macro 0.548 (few-shot 0.543).

## Files

- `index.html` — the page (UI, text normalization, result rendering)
- `worker.js` — loads the models and runs inference in a Web Worker
- `assets/` — logo, illustration, favicon

## Run locally

```bash
python -m http.server 8000
```

Then open http://localhost:8000. Qwen 3.5 needs a browser with WebGPU (recent Chrome or Edge).

## Data

[tasyapndya/free_nutritious_meals_sentiment_x_id_2025](https://huggingface.co/datasets/tasyapndya/free_nutritious_meals_sentiment_x_id_2025): 1,102 manually labelled tweets from X, January–March 2025.

Independent portfolio project, not affiliated with the Government of Indonesia or the National Nutrition Agency (Badan Gizi Nasional).
