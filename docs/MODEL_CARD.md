---
license: apache-2.0
language:
  - ar
  - en
library_name: onnxruntime
pipeline_tag: text-generation
tags:
  - customer-service
  - arabic
  - gulf-arabic
  - arabizi
  - intent-classification
  - onnx
  - int4
  - small-language-model
datasets:
  - HuggingFaceFW/fineweb-2
  - HuggingFaceFW/fineweb-edu
  - wikimedia/wikipedia
  - bitext/Bitext-customer-support-llm-chatbot-training-dataset
  - bitext/Bitext-retail-banking-llm-chatbot-training-dataset
  - bitext/Bitext-telco-llm-chatbot-training-dataset
  - PolyAI/banking77
  - clinc/clinc_oos
  - AmazonScience/massive
metrics:
  - accuracy
  - f1
---

# DanaLM (دانة)

DanaLM is a **62M-parameter decoder-only language model, trained from scratch for UAE customer
service**. It reads a customer's message in Gulf Arabic, English, Arabizi or a mix of them, and
answers with strict JSON: an intent from 21 labels and a short reply. It runs quantized on a CPU,
without PyTorch. It is meant to answer the easy messages on the device and to hand the rest to
a person or a bigger model.

Everything behind this card is in the repository: https://github.com/Mazharrony/danalm. That
includes the code, the configs, a decision log with every rule written before its result, the
data ledger, and the results page of every phase. All numbers below are measured.

## Model details

- **Architecture:** a Llama-style decoder.
  - 12 layers, d_model 640.
  - 10 attention heads sharing 2 key/value heads (grouped-query attention).
  - SwiGLU feed-forward layers (1,728 hidden), RMSNorm, and RoPE (θ = 10,000).
  - Tied input and output embeddings, a 1,024-token context, and 62.1M parameters.
- **Tokenizer:** `danalm-v1`, a 16,384-token byte-level BPE trained on Arabic, English and
  Arabizi. It needs 14% fewer tokens than Qwen3.5's tokenizer on Gulf Arabic.
- **Format:** `<|user|>message<|assistant|>{"intent": "...", "reply": "..."}<|endoftext|>`.
  - The message is normalized first, and personal data is masked (phone numbers, emails,
    Emirates IDs, card numbers, IBANs).
- **Confidence:** the chosen intent's share of the likelihoods of all 21 intents.
  - An answer is used on the device only when it is valid JSON, its reply is in the customer's
    language, and its confidence clears a threshold fixed on held-out real messages.
- **Deployed form:** an ONNX step graph with a key/value cache and 4-bit weights, 78 MB. The
  input embedding table stays in float32. The float32 (278 MB) and INT8 (101 MB) graphs exist too.
- **Author:** Mazhar Rony (MR). Trained on one RTX 4070 (12 GB).

## Intended use

- **For:** first-line triage of customer-service messages in banking, telecom and delivery. The
  model gives an intent and a draft reply, and escalates anything uncertain.
- **Also for:** research on small models for Gulf Arabic, and as a worked example of an honest
  training and evaluation process.
- **Intents (21):**
  - card_not_working, lost_or_stolen_card, balance_or_statement, transfer_issue,
    unrecognized_transaction, fees_and_charges, loans_and_credit;
  - bill_inquiry, network_or_internet_issue, plan_change, sim_or_number, roaming;
  - order_status, missing_or_wrong_item, cancel_order, refund_request,
    change_delivery_details, failed_delivery;
  - account_access, handoff_to_human, other.

**Out of scope:** acting on accounts, financial or legal advice, and any decision about a
customer without a person in the loop. The model cannot see accounts. Its replies are drafts;
they must not promise actions or state facts about a customer's case.

## Training data

All sources are free to use, and each licence was checked before download. Each source's size
and licence is in the [data ledger](DATA_LEDGER.md).

- **Pretraining: 1.49B tokens, two passes.**
  - Arabic, 64%:
    - FineWeb-2: 550M tokens of Standard Arabic and 300M of Najdi dialect (ODC-By-1.0).
    - Arabic Wikipedia: 100M (CC-BY-SA-3.0).
  - English, 36%:
    - FineWeb-Edu: 480M (ODC-By-1.0).
    - English Wikipedia: 50M (CC-BY-SA-3.0).
  - Customer-service text: Bitext support, banking and telco, 12.6M (CDLA-Sharing-1.0).
    Also CLINC150 and MASSIVE ar-SA training utterances (CC-BY).
- **Supervised fine-tuning: 34,302 examples.**
  - Most messages and replies were written by a local teacher, Qwen3.5-35B-A3B (Apache-2.0),
    in Gulf Arabic, English, Arabizi and mixed text. A blind label judge and a reply proofreader
    filtered them.
  - 2,921 are real English messages from the train splits of Banking77 (CC-BY-4.0) and CLINC150
    (CC-BY-3.0), mapped to these intents, with teacher-written replies.
  - Real out-of-scope questions from CLINC150 and MASSIVE were used for `other`.
  - 10,039 examples of question types the earlier data lacked: status checks, how-to, conditions,
    costs, other countries and off-topic requests.
    - 2,508 English messages from Bitext (CDLA-Sharing-1.0; its replies are not used).
    - 940 messages from MASSIVE en-US and ar-SA (CC-BY-4.0).
    - 6,591 Gulf Arabic, Arabizi and mixed messages written by the teacher.
    - Every reply was written by the teacher, under a rule to refer only to what the customer
      wrote.
  - Arabizi messages are answered in Arabic script.

## Evaluation

**Human test set, English part:** 64 real messages from the Banking77 and CLINC150 test splits,
labelled by the owner and never trained on. Each system was scored once.

| System | Intent accuracy (95% interval) | Valid JSON | Reply language |
|---|---:|---:|---:|
| **DanaLM, INT4 ONNX (deployed)** | **85.9%** (75–92%) | 100% | 100% |
| DanaLM, float32 | 84.4% (74–91%) | 100% | 100% |
| DanaLM, INT8 | 82.8% (72–90%) | 100% | 100% |
| The previous release (before the question-type data), INT4 | 84.4% (74–91%) | 98.4% | 96.9% |
| DanaLM before any real training data | 56.2% (44–68%) | 100% | 100% |
| CAMeLBERT-mix classifier, fine-tuned (110M) | 46.9% (35–59%) | – | – |
| Qwen3.5-35B-A3B, zero-shot | 89.1% (79–95%) | – | – |

- **Caveat:** the test messages come from the same two datasets as the real training messages,
  so part of the English gain is shared style.
- INT4 and float32 differ by one message here, and give a different intent on 5 of the 64.
  With 64 messages, this difference is noise.
- **Arabic is not covered.** The Gulf Arabic, Arabizi and mixed parts of the test set have not
  been written yet; they need a native speaker. Until then there is no human-labelled Arabic
  result.
- **Held-out real messages** (804, English, never trained on): 93.8% intent accuracy for INT4
  and 93.5% for float32.
- **Held-out question-type messages** (1,298 from Bitext and MASSIVE, English and Saudi
  Arabic): 91.8% for INT4 and 92.1% for float32. The previous model (D-033), unquantized,
  scored 65.4%. These come from the same datasets as a third of the new training data, so
  they show the new question types were learned rather than how well new wording is handled.
- **Synthetic validation split** (957 examples with teacher-written replies; 929 of the
  messages are teacher-written, 28 are real out-of-scope questions): 93.8% for INT4. Per
  variety (INT4 / float32): Gulf Arabic 94.8% / 94.8%, Arabizi 90.9% / 92.5%, mixed 94.4% /
  93.8%, English 94.9% / 94.9%, Saudi Arabic (11 messages) 100% / 90.9%. This split comes from
  the same teacher as the training data, so it overstates real-world accuracy.
- **Replies:** a Qwen judge rated the test replies on four questions: answers the question,
  polite and clear, right language, no invented facts or claimed actions.
  - This model: 87.5% (56/64) for INT4, and also 87.5% unquantized.
  - The previous model: 92.2% (59/64) unquantized. Its INT4 release scored 55 of its 63 valid
    replies (85.9% of the 64 messages).
  - The unquantized drop is within noise: 4 replies good only for the new model, 7 only for the
    old (p = 0.55).
  - On 300 development replies, the unquantized model scored 86.3%, against 79.0% for the
    previous model (p = 0.012).
- **Confidence:** at the threshold fixed on the real dev set, INT4's confidence clears it for
  90.6% of the test messages, and 89.7% of those are right. After the reply-language check, it
  answers 89.1% on the device, and 91.2% of those answers are right. The target was 95%.

**Speed and memory** (batch 1 on an Intel i9-13900K CPU, 4 threads, 100 development messages):
- The INT4 model takes **0.11 s per answer**, and 0.24 s including the confidence.
- The process peaks at 251 MB of memory.
- With 1 thread it takes 0.16 s per answer.
- The float32 PyTorch model without a cache takes 0.73 s per answer and 1,294 MB.

## Limitations and risks

- **Arabic quality is unverified by native speakers.** The Arabic and Arabizi results come
  from teacher-written data only.
- **Garbled replies.** Some replies mix scripts mid-sentence, mostly English replies to
  off-topic questions with Arabic words inside. The service escalates replies whose script does
  not fit the message instead of sending them. For INT4 that happens to 6 of 957, 6 of 804 and
  12 of 1,298 development replies, and to 1 of the 64 test replies. On the same 1,761
  development messages this model does it more often than the previous release (12 against 2);
  on the test set, less often (1 against 2).
- **Invented details.** Replies can invent reasons or policies ("usually blocked by the receiving
  bank"). About 1.5% of the English training replies say an action is done ("we have logged
  this"). Treat replies as drafts.
- **Known misreadings.** Questions about what the app can do abroad can get a wrong intent with
  high confidence. For example, "can i send money by bank app to Bangladesh?" is read as
  balance_or_statement at 0.73, just above the threshold.
- **Confidence** is not calibrated well enough on real messages to reach 95% accuracy on what
  it answers.
- **Coverage.** Only the 21 intents. Real training messages cover banking, everyday requests
  and 149 order-status questions; the telecom messages and the other delivery messages are
  synthetic. Messages outside these domains should be routed to `other` and escalated.
- **Bias.** Most messages were written by one teacher model, whose style and assumptions the
  model inherits.

## How to use

The inference code needs only numpy, tokenizers and onnxruntime, not PyTorch:

```python
from danalm.infer.predictor import Predictor

predictor = Predictor("int4", threads=4)  # a folder with model.onnx, tokenizer.json, danalm.json
predictor.predict("وين طلبي؟ صار له ساعتين وما وصل")
# {'intent': 'order_status', 'reply': '...', 'confidence': ..., 'route': 'on_device',
#  'valid_json': True, 'reply_fits': True, 'message_masked': '...', 'latency_ms': ...}
```

The repository also has a FastAPI service (`POST /predict`) and a Dockerfile. An in-browser demo
runs this INT4 model with ONNX Runtime Web: https://huggingface.co/spaces/Mazharrony/danalm-demo

## Training compute

- **Pretraining:** two passes of 5,675 steps (1.49B tokens each) on one RTX 4070 capped at
  200 W, at about 87k tokens/s. The first pass took 4 h 49 min; the second was the same length.
- **Fine-tuning:** about 20 minutes per run (34,302 examples, 5 epochs); 4 runs in the final
  round.
- **Teacher data:** generated locally with llama.cpp.

## Citation

```bibtex
@misc{danalm2026,
  title  = {DanaLM: a small language model for UAE customer service, trained from scratch},
  author = {Rony, Mazhar},
  year   = {2026},
  url    = {https://github.com/Mazharrony/danalm}
}
```
