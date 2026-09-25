"""DanaLM demo (Phase 7, D-034): intent, reply, confidence and route for a UAE customer-service
message, on the CPU with ONNX Runtime. No PyTorch.

Locally:  uv run --group demo python scripts/build_space.py --config configs/deploy/phase7.yaml
          then  cd artifacts/space && uv run --group demo python app.py
DANALM_MODEL_DIR (default "model") holds model.onnx, tokenizer.json and danalm.json.
"""

import os

import gradio as gr

from danalm.infer.predictor import MessageTooLong, Predictor

predictor = Predictor(
    os.environ.get("DANALM_MODEL_DIR", "model"), threads=int(os.environ.get("DANALM_THREADS", "2"))
)
EXAMPLES = [  # written for the demo; none is a test-set message
    "my card got stuck in the ATM machine this morning",
    "وين طلبي؟ صار له ساعتين وما وصل",
    "3andi mushkila fil card, ma yishtaghil",
    "I want to talk to a real person please",
    "how much do you charge for sending money to India?",
    "ابي اغير رقم التوصيل للطلب",
]
ABOUT = f"""
**DanaLM** is a 62M-parameter language model trained from scratch for UAE customer service. It
reads Gulf Arabic, English, Arabizi or a mix, and answers with an intent and a short reply. This
demo runs the **{predictor.meta.get("variant", "?")}** ONNX model on a CPU.

- **Route:** the model answers on the device when its answer is valid and its confidence is at
  least {predictor.threshold:.2f} (fixed on held-out real messages). Otherwise the message would go
  to a person or a bigger model.
- **Masked message:** phone numbers, emails, Emirates IDs, card numbers and IBANs are masked
  before the model sees the text. Only the masked text would leave the device.
- **Limits:** English was checked on 64 real messages; the numbers are in the repository. The
  Gulf Arabic, Arabizi and mixed parts have not been checked by a native speaker yet. The replies
  come from a small model and can be wrong. Do not enter real personal data.
"""


def answer(message: str) -> tuple[str, str, str, str, str, str]:
    if not message.strip():
        return "", "", "", "", "", ""
    try:
        out = predictor.predict(message)
    except MessageTooLong:
        return "", "The message is too long for this demo.", "", "", "", ""
    route = "answer on the device" if out["route"] == "on_device" else "escalate"
    return (out["intent"] or "(no valid answer)", out["reply"] or "", f"{out['confidence']:.2f}",
            route, out["message_masked"], f"{out['latency_ms']:.0f} ms")  # fmt: skip


with gr.Blocks(title="DanaLM") as demo:
    gr.Markdown("# DanaLM (دانة)\nCustomer-service intent and reply, on the CPU.")
    with gr.Row():
        with gr.Column():
            message = gr.Textbox(label="Customer message", lines=3)
            send = gr.Button("Answer", variant="primary")
            gr.Examples(EXAMPLES, inputs=message)
        with gr.Column():
            intent = gr.Textbox(label="Intent")
            reply = gr.Textbox(label="Reply", lines=3)
            with gr.Row():
                confidence = gr.Textbox(label="Confidence")
                route = gr.Textbox(label="Route")
                latency = gr.Textbox(label="Time")
            masked = gr.Textbox(label="Masked message (what may leave the device)")
    gr.Markdown(ABOUT)
    outputs = [intent, reply, confidence, route, masked, latency]
    send.click(answer, inputs=message, outputs=outputs)
    message.submit(answer, inputs=message, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
