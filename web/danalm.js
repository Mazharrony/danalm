// DanaLM in the browser (Phase 8): a port of the torch-free Python inference path.
// - normalize / maskPii / detectLang / replyFits: danalm/text.py, with Python's Unicode \w \d \s \b
//   written out, so both give the same result;
// - Tokenizer: the byte-level BPE of tokenizer.json (Split regex, ByteLevel, merges, added tokens);
// - Predictor: danalm/infer (KV-cache greedy answer, the 21-intent confidence, the D-035 guard)
//   over the ONNX step graph with ONNX Runtime Web.
// scripts/web_parity.py and web/parity.html compare every step with Python on development messages.

// ---------------------------------------------------------------- text (danalm/text.py)
const W = "\\p{L}\\p{N}_"; // Python's Unicode \w, inside a class
const D = "\\p{Nd}"; // Python's Unicode \d
const S = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000"; // str.isspace
const B = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`; // Python's Unicode \b
const re = (src, flags = "gu") => new RegExp(src, flags);

const AR_DIACRITICS = re("[\\u0610-\\u061a\\u064b-\\u065f\\u0670\\u06d6-\\u06ed]");
const ALEF_VARIANTS = re("[\\u0625\\u0623\\u0622\\u0671]");
const URL = re(`https?://[^${S}]+|www\\.[^${S}]+`);
const EMAIL = re(`${B}[${W}.+\\-]+@[${W}\\-]+\\.[${W}.\\-]+${B}`);
const EMIRATES_ID = re(`(?<!${D})784[\\-${S}]?${D}{4}[\\-${S}]?${D}{7}[\\-${S}]?${D}(?!${D})`);
const IBAN = re(`(?<![A-Za-z])AE${D}{2}(?:[ \\-]?${D}){19}(?!${D})`, "giu");
const CARD = re(
  `(?<!${D})(?:${D}{4}[ \\-]?){3}${D}{4}(?!${D})` +
    `|(?<!${D})${D}{4}[ \\-]?${D}{6}[ \\-]?${D}{5}(?!${D})` +
    `|(?<!${D})(?!(?:00)?971)${D}{13,19}(?!${D})`
);
const PHONE = re(
  `(?<!${D})(?:\\+|00)?971[${S}\\-]?${D}{1,2}[${S}\\-]?${D}{3}[${S}\\-]?${D}{4}(?!${D})` +
    `|(?<!${D})0?5${D}[${S}\\-]?${D}{3}[${S}\\-]?${D}{4}(?!${D})` +
    `|(?<!${D})0[2-4679][${S}\\-]?${D}{3}[${S}\\-]?${D}{4}(?!${D})`
);
const LONG_NUMBER = re(`(?<!${D})${D}{9,}(?!${D})`);
const PLACEHOLDER = re("<(?:URL|EMAIL|EID|IBAN|CARD|PHONE|NUM)>");
const REPEAT = re(`([^${D}${S}])\\1{4,}`);
const WS = re(`[${S}]+`);
const EDGE_WS = re(`^[${S}]+|[${S}]+$`);
const AR_CHAR = re("[\\u0600-\\u06ff\\u0750-\\u077f\\ufb50-\\ufdff\\ufe70-\\ufeff]");
const LAT_CHAR = re("[A-Za-z]");
const ARABIZI_HINT = re(
  `^(?:${B}[${W}]*[a-zA-Z][2356789][a-zA-Z][${W}]*${B}|${B}[2356789][a-zA-Z]{2,}${B})$`,
  "u"
);
const LATIN_WORD = re("[A-Za-z0-9']+");
const EN_NUMERIC = re(
  `${B}${D}+(?:st|nd|rd|th|am|pm|fa|ds|gb|mb|kg|km|min|mins|hr|hrs|aed|dhs|usd)${B}`,
  "giu"
);
const ARABIZI_WORDS = new Set(
  ("abga abgha abi alheen alhin fawran ghalat habibi hada haka hatha inshallah jadid jdid jiddan " +
    "khalas kthir laish leish liya mafi mashallah mashi shlon shlonak shlonich shu shukran waayed " +
    "wain wala walla wallah waqt wayed yalla ykoon ykoun zain zein").split(" ")
);
const ARABIZI_MIN_SHARE = 0.1;
const LANG_TO_VARIETY = { en: "english", ar: "gulf_arabic", arabizi: "arabizi", mixed: "mixed" };

export function maskPii(text) {
  text = text.replace(/[٠-٩]/g, (c) => String(c.charCodeAt(0) - 0x0660));
  text = text.replace(/[۰-۹]/g, (c) => String(c.charCodeAt(0) - 0x06f0));
  text = text.replace(URL, "<URL>").replace(EMAIL, "<EMAIL>").replace(EMIRATES_ID, "<EID>");
  text = text.replace(IBAN, "<IBAN>").replace(CARD, "<CARD>").replace(PHONE, "<PHONE>");
  return text.replace(LONG_NUMBER, "<NUM>");
}

export function normalize(text, { strip_diacritics, unify_alef }) {
  text = text.normalize("NFKC").replaceAll("ـ", "");
  if (strip_diacritics) text = text.replace(AR_DIACRITICS, "");
  if (unify_alef) text = text.replace(ALEF_VARIANTS, "ا").replaceAll("ى", "ي");
  text = maskPii(text);
  text = text.replace(REPEAT, (_, c) => c.repeat(3));
  return text.replace(WS, " ").replace(EDGE_WS, "");
}

const count = (regex, text) => (text.match(regex) || []).length;

export function detectLang(text) {
  text = text.replace(PLACEHOLDER, " ");
  const ar = count(AR_CHAR, text);
  const lat = count(LAT_CHAR, text);
  if (ar + lat === 0) return "other";
  const ratio = ar / (ar + lat);
  if (ratio > 0.85) return "ar";
  if (ratio < 0.15) {
    const words = text.replace(EN_NUMERIC, " ").match(LATIN_WORD) || [];
    const evidence = words.filter((w) => ARABIZI_HINT.test(w) || ARABIZI_WORDS.has(w.toLowerCase()));
    return words.length && evidence.length / words.length >= ARABIZI_MIN_SHARE ? "arabizi" : "en";
  }
  return "mixed";
}

export function replyFits(message, reply, replyLangs) {
  const lang = detectLang(message);
  if (!(lang in LANG_TO_VARIETY)) return true;
  if (lang === "en" && re("[\\u0600-\\u06ff\\u0750-\\u077f\\ufb50-\\ufdff\\ufe70-\\ufeff]", "u").test(reply.replace(PLACEHOLDER, " "))) return false;
  return replyLangs[LANG_TO_VARIETY[lang]].includes(detectLang(reply));
}

// ---------------------------------------------------------------- tokenizer (tokenizer.json)
function bytesToUnicode() {
  // GPT-2's reversible map from bytes to printable characters (the ByteLevel alphabet)
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const enc = new Array(256);
  const dec = new Map();
  bs.forEach((b, i) => {
    enc[b] = String.fromCodePoint(cs[i]);
    dec.set(enc[b], b);
  });
  return [enc, dec];
}

// the Split pattern of tokenizer.json; JavaScript has no (?i:...) group, so the contractions
// are spelled out case by case, in the same order
const SPLIT =
  "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]" +
  "|[^\\r\\n\\p{L}\\p{M}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*" +
  "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

export class Tokenizer {
  constructor(json) {
    this.vocab = json.model.vocab;
    this.ranks = new Map(json.model.merges.map((m, i) => [Array.isArray(m) ? m.join(" ") : m, i]));
    this.idToToken = [];
    for (const [t, id] of Object.entries(this.vocab)) this.idToToken[id] = t;
    this.added = new Map();
    for (const a of json.added_tokens) {
      this.added.set(a.content, a.id);
      this.idToToken[a.id] = a.content;
    }
    const alts = [...this.added.keys()].sort((a, b) => b.length - a.length);
    this.addedRe = new RegExp(alts.map((a) => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
    this.splitRe = new RegExp(SPLIT, "gu");
    [this.byteEnc, this.byteDec] = bytesToUnicode();
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  tokenToId(token) {
    return this.added.get(token) ?? this.vocab[token];
  }

  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = [...word];
    while (parts.length > 1) {
      let best = Infinity;
      let pair = null;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < best) {
          best = r;
          pair = [parts[i], parts[i + 1]];
        }
      }
      if (!pair) break;
      const merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === pair[0] && parts[i + 1] === pair[1]) {
          merged.push(pair[0] + pair[1]);
          i++;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    this.cache.set(word, parts);
    return parts;
  }

  encodePlain(text, ids) {
    // Split with "Isolated" behaviour: every match and every gap between matches is a piece
    let last = 0;
    const pieces = [];
    for (const m of text.matchAll(this.splitRe)) {
      if (m.index > last) pieces.push(text.slice(last, m.index));
      pieces.push(m[0]);
      last = m.index + m[0].length;
    }
    if (last < text.length) pieces.push(text.slice(last));
    for (const piece of pieces) {
      const chars = Array.from(this.utf8.encode(piece), (b) => this.byteEnc[b]).join("");
      for (const t of this.bpe(chars)) {
        const id = this.vocab[t];
        if (id === undefined) throw new Error(`token not in the vocabulary: ${t}`);
        ids.push(id);
      }
    }
  }

  encode(text) {
    // added tokens (e.g. <PHONE>) are matched first, as in tokenizers
    const ids = [];
    let last = 0;
    for (const m of text.matchAll(this.addedRe)) {
      if (m.index > last) this.encodePlain(text.slice(last, m.index), ids);
      ids.push(this.added.get(m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) this.encodePlain(text.slice(last), ids);
    return ids;
  }

  decode(ids) {
    // ByteLevel decoding of every token (added tokens included), then lossy UTF-8
    const bytes = [];
    for (const id of ids) {
      for (const ch of this.idToToken[id]) {
        const b = this.byteDec.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }
}

// ---------------------------------------------------------------- the answer format
const INTENT_PREFIX = '{"intent": "';

export function parseAnswer(text, intents) {
  let obj;
  try {
    obj = JSON.parse(text);
  } catch {
    return { parsed: false, valid: false, intent: null, reply: null };
  }
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    return { parsed: true, valid: false, intent: null, reply: null };
  }
  const intent = typeof obj.intent === "string" ? obj.intent : null;
  const reply = typeof obj.reply === "string" ? obj.reply : null;
  const keys = Object.keys(obj);
  const valid =
    keys.length === 2 && keys.includes("intent") && keys.includes("reply") && intents.has(intent) && reply !== null;
  return { parsed: true, valid, intent, reply };
}

// ---------------------------------------------------------------- the model (danalm/infer)
function logSoftmaxAt(logits, offset, size) {
  let max = -Infinity;
  for (let i = 0; i < size; i++) max = Math.max(max, logits[offset + i]);
  let sum = 0;
  for (let i = 0; i < size; i++) sum += Math.exp(logits[offset + i] - max);
  return max + Math.log(sum); // log-sum-exp: logprob(i) = logits[i] - this
}

function argmaxAt(logits, offset, size) {
  let best = 0;
  for (let i = 1; i < size; i++) if (logits[offset + i] > logits[offset + best]) best = i;
  return best;
}

export class Predictor {
  static async create(ort, files) {
    // files: {model: ArrayBuffer, tokenizer: object, meta: object}
    const p = new Predictor();
    p.ort = ort;
    p.meta = files.meta;
    p.session = await ort.InferenceSession.create(files.model, { executionProviders: ["wasm"] });
    p.tok = new Tokenizer(files.tokenizer);
    const sp = p.meta.special;
    p.chat = Object.fromEntries(Object.entries(sp).map(([k, v]) => [k, p.tok.tokenToId(v)]));
    p.intents = p.meta.intents;
    p.intentSet = new Set(p.intents);
    p.prefix = p.tok.encode(INTENT_PREFIX);
    p.conts = p.intents.map((name) => p.tok.encode(`${INTENT_PREFIX}${name}",`).slice(p.prefix.length));
    const cfg = p.meta.model_config;
    p.kv = cfg.n_kv_heads;
    p.hd = cfg.d_model / cfg.n_heads;
    p.vocab = cfg.vocab_size;
    p.layers = cfg.n_layers;
    return p;
  }

  emptyPast(batch) {
    return Array.from({ length: 2 * this.layers }, () => new this.ort.Tensor("float32", new Float32Array(0), [batch, this.kv, 0, this.hd]));
  }

  async step(ids, batch, positions, past) {
    const t = positions.length;
    const feeds = {
      input_ids: new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [batch, t]),
      positions: new this.ort.Tensor("int64", BigInt64Array.from(positions, BigInt), [t]),
    };
    const names = this.session.inputNames.slice(2);
    names.forEach((n, i) => (feeds[n] = past[i]));
    const out = await this.session.run(feeds);
    const present = this.session.outputNames.slice(1).map((n) => out[n]);
    return [out[this.session.outputNames[0]].data, present];
  }

  repeatPast(past, n) {
    // each (1, kv, len, hd) cache copied n times along the batch
    return past.map((p) => {
      const data = new Float32Array(p.data.length * n);
      for (let i = 0; i < n; i++) data.set(p.data, i * p.data.length);
      return new this.ort.Tensor("float32", data, [n, ...p.dims.slice(1)]);
    });
  }

  async predict(message) {
    const start = performance.now();
    const norm = this.meta.normalize;
    const masked = normalize(message, norm);
    const prompt = [this.chat.user, ...this.tok.encode(masked), this.chat.assistant];
    const maxNew = this.meta.max_new_tokens;
    if (prompt.length + maxNew > this.meta.model_config.max_seq_len) throw new Error("message too long");
    const V = this.vocab;
    // the prompt, then greedy decoding with the cache (danalm/infer/decode.py)
    let [logits, past] = await this.step(prompt, 1, [...prompt.keys()], this.emptyPast(1));
    const promptPast = past;
    const generated = [];
    let next = argmaxAt(logits, (prompt.length - 1) * V, V);
    let finished = false;
    for (let n = 0; n < maxNew; n++) {
      if (next === this.chat.eos) {
        finished = true;
        break;
      }
      generated.push(next);
      if (n === maxNew - 1) break;
      [logits, past] = await this.step([next], 1, [prompt.length + n], past);
      next = argmaxAt(logits, 0, V);
    }
    const answer = parseAnswer(this.tok.decode(generated), this.intentSet);
    // the 21 intent scores from the prompt's cache (label_logprobs_from)
    const at = prompt.length;
    const [pl, pp] = await this.step(this.prefix, 1, this.prefix.map((_, i) => at + i), promptPast);
    const off = (this.prefix.length - 1) * V;
    const lse = logSoftmaxAt(pl, off, V);
    const scores = this.conts.map((c) => pl[off + c[0]] - lse);
    const longest = Math.max(...this.conts.map((c) => c.length));
    if (longest > 1) {
      const n = this.conts.length;
      const inp = [];
      for (const c of this.conts) for (let j = 0; j < longest - 1; j++) inp.push(j < c.length - 1 ? c[j] : this.chat.pad);
      const pos = Array.from({ length: longest - 1 }, (_, j) => at + this.prefix.length + j);
      const [cl] = await this.step(inp, n, pos, this.repeatPast(pp, n));
      this.conts.forEach((c, i) => {
        for (let j = 1; j < c.length; j++) {
          const o = (i * (longest - 1) + (j - 1)) * V;
          scores[i] += cl[o + c[j]] - logSoftmaxAt(cl, o, V);
        }
      });
    }
    const max = Math.max(...scores);
    const exps = scores.map((s) => Math.exp(s - max));
    const total = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / total);
    const conf = answer.valid ? probs[this.intents.indexOf(answer.intent)] : 0;
    const fits = answer.valid && replyFits(masked, answer.reply, this.meta.reply_langs);
    const onDevice = fits && conf >= this.meta.threshold;
    return {
      intent: answer.valid ? answer.intent : null,
      reply: answer.valid ? answer.reply : null,
      confidence: Math.round(conf * 1e4) / 1e4,
      route: onDevice ? "on_device" : "escalate",
      valid_json: answer.valid,
      reply_fits: fits,
      finished,
      message_masked: masked,
      latency_ms: Math.round((performance.now() - start) * 10) / 10,
      generated_ids: generated,
    };
  }
}
