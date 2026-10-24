// Inference worker: runs IndoBERT-lite (WASM) and Qwen3.5-2B (WebGPU) off the main thread.
import {
    pipeline,
    env,
    AutoTokenizer,
    Qwen3_5ForCausalLM,
    LogitsProcessor,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

env.allowLocalModels = false;

// Label order used everywhere: index 0..2
const LABELS = ['negatif', 'netral', 'positif'];

// Same prompt and few-shot examples as the notebook (outputs/llm_prompt_config.json)
const SYSTEM_PROMPT =
    'Anda adalah analis sentimen opini publik berbahasa Indonesia. Tugas Anda: tentukan sentimen sebuah tweet tentang program Makan Bergizi Gratis (MBG).\n\n' +
    'Definisi label:\n' +
    '- positif: mendukung, memuji, berterima kasih, atau optimis terhadap program MBG dan pelaksananya.\n' +
    '- negatif: mengkritik, menolak, menyindir/sarkas, atau menyoroti masalah (misalnya keracunan, makanan basi, anggaran, korupsi, salah sasaran).\n' +
    '- netral: menyampaikan informasi, berita, pertanyaan, atau opini tanpa sikap mendukung maupun menolak yang jelas.\n\n' +
    'Jawab HANYA dengan satu kata: positif, negatif, atau netral.';
const FEW_SHOTS = [
    ['makan bergizi gratis kenapa tidak diambil dari dana desa dan penanggung jawabnya kades masing masing sederhana tapi tidak dilakukan', 'negatif'],
    ['ratusan siswa ini turun ke jalan tolak makan bergizi gratis', 'negatif'],
    ['jadi bagian dari solusi dukung program makan bergizi gratis untuk anak anak', 'positif'],
    ['polres indragiri hulu mengadakan makan bergizi gratis di tk kemala bhayangkari 4 iqbal 91 polresinhu kapolresinhu fahriansalehsiregar', 'netral'],
    ['yuk bantu anak anak dapat gizi sesuai kebutuhan dengan makan bergizi gratis', 'positif'],
    ['presiden prabowo subianto melakukan kunjungan mendadak ke dua sekolah di jakarta timur untuk meninjau pelaksanaan program makan bergizi gratis mbg pada senin 3 2 2025 pagi n prabowosubianto makanbergizigratis jaktim', 'netral'],
];

const QWEN_ID = 'onnx-community/Qwen3.5-2B-ONNX-OPT';

const state = { bert: null, qwen: null, qwenTok: null, qwenLabelIds: null };

// Qwen chat format, identical to the notebook's apply_chat_template(..., enable_thinking=False)
function buildQwenPrompt(text) {
    const turn = (role, content) => `<|im_start|>${role}\n${content}<|im_end|>\n`;
    let prompt = turn('system', SYSTEM_PROMPT);
    for (const [shot, label] of FEW_SHOTS) {
        prompt += turn('user', `Tweet: ${shot}\nSentimen:`) + turn('assistant', label);
    }
    prompt += turn('user', `Tweet: ${text}\nSentimen:`);
    return prompt + '<|im_start|>assistant\n<think>\n\n</think>\n\n';
}

// Sums byte progress over all files of one model so the UI can show a single bar
function progressReporter(model) {
    const files = new Map();
    return (p) => {
        if (p.status !== 'progress' || !p.file) return;
        files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
        let loaded = 0, total = 0;
        for (const f of files.values()) { loaded += f.loaded; total += f.total; }
        postMessage({ type: 'progress', model, loaded, total });
    };
}

async function loadBert(modelId) {
    if (state.bert) return;
    state.bert = await pipeline('text-classification', modelId, {
        dtype: 'fp32',
        device: 'wasm',
        progress_callback: progressReporter('indobert'),
    });
}

async function loadQwen() {
    if (state.qwen) return;
    const adapter = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null;
    if (!adapter) throw new Error('WEBGPU_UNAVAILABLE');
    const dtype = adapter.features.has('shader-f16') ? 'q4f16' : 'q4';
    const progress_callback = progressReporter('qwen');
    state.qwenTok = await AutoTokenizer.from_pretrained(QWEN_ID, { progress_callback });
    state.qwen = await Qwen3_5ForCausalLM.from_pretrained(QWEN_ID, {
        dtype: { embed_tokens: dtype, decoder_model_merged: dtype },
        device: 'webgpu',
        progress_callback,
    });
    // First sub-token of each label ("neg", "net", "posit") — they differ, so one step decides the label
    state.qwenLabelIds = LABELS.map((l) => state.qwenTok.encode(l, { add_special_tokens: false })[0]);
}

// Keeps the next-token logits of the first generation step
class CaptureLogits extends LogitsProcessor {
    _call(input_ids, logits) {
        this.logits = logits;
        return logits;
    }
}

function softmax(xs) {
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
}

async function classify(model, text) {
    if (model === 'indobert') {
        const out = await state.bert(text, { top_k: null });
        return LABELS.map((l) => out.find((o) => o.label === l)?.score ?? 0);
    }
    const inputs = state.qwenTok(buildQwenPrompt(text), { add_special_tokens: false });
    const capture = new CaptureLogits();
    await state.qwen.generate({
        ...inputs,
        max_new_tokens: 1,
        do_sample: false,
        repetition_penalty: 1.0,
        logits_processor: [capture],
    });
    const row = capture.logits[0].data;
    return softmax(state.qwenLabelIds.map((id) => row[id]));
}

self.onmessage = async ({ data }) => {
    const { type, model, id } = data;
    try {
        if (type === 'load') {
            const t0 = performance.now();
            if (model === 'indobert') await loadBert(data.modelId);
            else await loadQwen();
            postMessage({ type: 'ready', model, ms: performance.now() - t0 });
        } else if (type === 'classify') {
            const t0 = performance.now();
            const probs = await classify(model, data.text);
            postMessage({ type: 'result', model, id, probs, ms: performance.now() - t0 });
        }
    } catch (err) {
        postMessage({ type: 'error', model, id, message: String(err?.message ?? err) });
    }
};
