---
title: DanaLM
emoji: 🐚
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: A 62M customer-service model running in your browser
custom_headers:                # cross-origin isolation, so ONNX Runtime Web can use threads
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# DanaLM, in your browser

A 62M-parameter decoder trained from scratch for UAE customer service. It reads Gulf Arabic,
English, Arabizi or a mix, and returns an intent, a short reply, a confidence and a route: answer
on the device, or escalate.

This Space is a static page. Your browser downloads the 4-bit ONNX model from
[Mazharrony/danalm](https://huggingface.co/Mazharrony/danalm) once and runs it with ONNX Runtime
Web. Nothing you type is sent anywhere.

- `danalm.js` is a port of the Python inference code: text normalization and PII masking, the
  byte-level BPE tokenizer, KV-cache decoding, the 21-intent confidence, and the reply-language
  guard.
- The repository checks the port against Python on development messages (`web/parity.html`).
- Code, data sources, decisions and every result: https://github.com/Mazharrony/danalm
