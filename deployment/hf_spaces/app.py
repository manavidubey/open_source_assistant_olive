"""
Hugging Face Spaces Deployment — Gradio App

Lightweight deployment of the OSS model (Qwen 2.5) on HF Spaces.
This file is designed to be the entry point for a Gradio-based Space.
"""

import os
import gradio as gr
from huggingface_hub import InferenceClient

# ── Configuration ────────────────────────────────────────────────
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI personal assistant. "
    "You provide accurate, well-reasoned answers. If you don't know "
    "something, you say so. You refuse harmful or unethical requests."
)

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN or None)


def respond(message: str, history: list[dict]) -> str:
    """Generate a response with conversation history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# ── Gradio Interface ─────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🤖 AI Personal Assistant (OSS — Qwen 2.5)",
    description="Open-source personal assistant powered by Qwen 2.5. Supports multi-turn conversations with memory.",
    examples=[
        "What is the capital of Australia?",
        "Explain quantum computing in simple terms.",
        "Help me plan a weekend trip to Paris.",
        "What's the difference between Python and JavaScript?",
    ],
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
    fill_height=True,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
