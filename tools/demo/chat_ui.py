import os
import time

import gradio as gr
import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from mlx_lm import load

os.environ["TQ_USE_METAL"] = "1"

model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
print(f"Loading {model_name}...")
model, tokenizer = load(model_name)

# Custom CSS for a ChatGPT-like clean UI
css = """
body, .gradio-container {background-color: #343541 !important;}
textarea {background-color: #40414f !important; color: #ececf1 !important; outline: none !important; border: 1px solid #565869 !important; border-radius: 8px !important;}
label, h2, h3 {color: #ececf1 !important;}
.stats-box {background-color: #202123; padding: 15px; border-radius: 8px; color: #ececf1 !important; font-family: monospace; border: 1px solid #565869;}
.message-wrap p {color: #111827 !important; font-size: 15px; font-weight: 500;}
"""


def user_action(user_message, history):
    hf = history or []
    is_v5_plus = int(getattr(gr, "__version__", "0").split(".")[0]) >= 5

    if is_v5_plus:
        return "", hf + [{"role": "user", "content": user_message}]

    return "", hf + [[user_message, None]]


def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type", "") == "text"
        )
    return str(content)


def bot_action(history, max_tokens, temperature, k_bits, group_size):
    if not history:
        yield history, "<div class='stats-box'>Waiting for generation...</div>"
        return

    messages = []
    for m in history[:-1]:
        if isinstance(m, dict):
            text_content = extract_text(m.get("content", ""))
            if text_content:
                messages.append(
                    {"role": m.get("role", "user"), "content": text_content}
                )
        else:
            if m[0]:
                messages.append({"role": "user", "content": extract_text(m[0])})
            if m[1]:
                messages.append({"role": "assistant", "content": extract_text(m[1])})

    # Safely get last user message
    last_item = history[-1]
    last_user_msg = (
        extract_text(last_item.get("content", ""))
        if isinstance(last_item, dict)
        else extract_text(last_item[0])
    )
    messages.append({"role": "user", "content": last_user_msg})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    is_dict_format = isinstance(history[-1], dict)
    if is_dict_format:
        history.append({"role": "assistant", "content": ""})
    else:
        history[-1] = [last_user_msg, ""]

    sampler = make_sampler(temp=float(temperature))

    generator = stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
        sampler=sampler,
        turboquant_k_start=0,
        turboquant_main_bits=int(k_bits),
        turboquant_group_size=int(group_size),
    )

    start_time = time.time()
    tokens = 0

    for response in generator:
        if response.text:
            if is_dict_format:
                history[-1]["content"] += response.text
            else:
                # Need to replace the tuple/list since tuples are immutable, lists are fine
                history[-1] = [history[-1][0], history[-1][1] + response.text]

            tokens += 1
            elapsed = time.time() - start_time
            tps = tokens / elapsed if elapsed > 0 else 0

            try:
                mem_gb = mx.metal.get_active_memory() / (1024**3)
                mem_format = f"{mem_gb:.2f} GB"
            except AttributeError:
                mem_format = "N/A"

            stats = f"<div class='stats-box'>⚡ <b>Speed:</b> {tps:.2f} tokens/s<br>🧠 <b>Memory:</b> {mem_format}<br>📝 <b>Tokens:</b> {tokens}</div>"

            yield history, stats


with gr.Blocks(fill_height=True) as demo:
    gr.HTML(
        "<h2 style='text-align: center; font-family: system-ui;'>TurboQuant ✨ Llama-3.2</h2>"
    )

    with gr.Row():
        # Left Sidebar for config and stats
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                gr.Markdown("### ⚙️ Hardware Settings")
                max_tokens = gr.Slider(
                    minimum=10, maximum=1024, value=512, step=1, label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature"
                )
                k_bits = gr.Slider(
                    minimum=2, maximum=8, value=3, step=1, label="KV Cache k_bits"
                )
                group_size = gr.Dropdown(
                    choices=[32, 64, 128], value=64, label="KV Cache Group Size"
                )

            gr.Markdown("### 📊 Live Performance")
            stats_box = gr.HTML(
                value="<div class='stats-box'>⚡ <b>Speed:</b> 0.00 tokens/s<br>🧠 <b>Memory:</b> 0.00 GB<br>📝 <b>Tokens:</b> 0</div>"
            )

        # Main Chat Area
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(scale=1, container=False, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Message Llama-3.2...",
                    container=False,
                    scale=8,
                )
                submit = gr.Button("Send 🚀", scale=1, variant="primary")

    # Wiring interactions
    msg.submit(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action,
        [chatbot, max_tokens, temperature, k_bits, group_size],
        [chatbot, stats_box],
    )
    submit.click(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_action,
        [chatbot, max_tokens, temperature, k_bits, group_size],
        [chatbot, stats_box],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, css=css, theme=gr.themes.Monochrome())
