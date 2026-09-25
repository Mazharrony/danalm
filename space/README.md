---
title: DanaLM
emoji: 🐚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: A 62M model for UAE customer service, on the CPU
---

# DanaLM demo

A 62M-parameter decoder trained from scratch for UAE customer service. It reads Gulf Arabic,
English, Arabizi or a mix, and returns an intent, a short reply, a confidence and a route
(answer on the device, or escalate). It runs with ONNX Runtime on the CPU, without PyTorch.

Code, data sources, decisions and every result: https://github.com/Mazharrony/danalm

This folder is assembled by `scripts/build_space.py` in that repository: the app, the torch-free
`danalm` inference modules and the deployed model (`model/`).
