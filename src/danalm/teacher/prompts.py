"""Prompts shared by several scripts."""


def intent_judge_prompt(intents: list[dict], messages: list[str]) -> str:
    """The blind label-judge prompt of Phase 2 (scripts/verify_sft.py), also the zero-shot
    teacher baseline of Phase 6 (D-032): the intents with their descriptions, then the numbered
    messages. The judge never sees an intended label."""
    labels = "\n".join(f"- {i['name']}: {i['description']}" for i in intents)
    numbered = "\n".join(f"{n}. {m}" for n, m in enumerate(messages, start=1))
    return (
        f"Intents:\n{labels}\n\nMessages:\n{numbered}\n\n"
        "For each message, answer with its number and the single best intent name, "
        'like "3: order_status".'
    )
