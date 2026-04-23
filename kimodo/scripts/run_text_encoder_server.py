# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import tempfile

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kimodo.model import resolve_target

from .gradio_theme import get_gradio_theme

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
DEFAULT_TEXT = "A person walks and falls to the ground."
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 9550
DEFAULT_TMP_FOLDER = os.path.join(tempfile.gettempdir(), "text_encoder")
DEFAULT_TEXT_ENCODER = "llm2vec"
DEFAULT_ENABLE_GENERATION_MODEL = False
DEFAULT_GENERATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_GENERATION_QUANTIZATION = "4bit"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16",
            "llm_dim": 4096,
            "device": "auto",
        },
        "display_name": "LLM2Vec",
    }
}


class DemoWrapper:
    def __init__(
        self,
        text_encoder,
        tmp_folder,
        *,
        generation_tokenizer=None,
        generation_model=None,
    ):
        self.text_encoder = text_encoder
        self.tmp_folder = tmp_folder
        self.generation_tokenizer = generation_tokenizer
        self.generation_model = generation_model

    def _generate_with_shared_llm(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        if self.generation_tokenizer is not None and self.generation_model is not None:
            tokenizer = self.generation_tokenizer
            llm_model = self.generation_model

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_len = int(encoded["input_ids"].shape[1]) if "input_ids" in encoded else 0

            model_device = next(llm_model.parameters()).device
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

            do_sample = float(temperature) > 1e-6
            generate_kwargs = {
                "max_new_tokens": max(8, int(max_new_tokens)),
                "do_sample": do_sample,
                "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
            }
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = int(eos_token_id)
            if do_sample:
                generate_kwargs["temperature"] = float(max(temperature, 1e-6))
                generate_kwargs["top_p"] = float(min(1.0, max(0.1, top_p)))

            with torch.no_grad():
                output = llm_model.generate(**encoded, **generate_kwargs)

            generated_ids = output[0, input_len:] if input_len > 0 and output.shape[1] > input_len else output[0]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not decoded:
                full_decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
                if full_decoded.lower().startswith(prompt.lower()):
                    full_decoded = full_decoded[len(prompt) :].strip()
                decoded = full_decoded
            if not decoded:
                raise RuntimeError("Generation model returned empty text")
            return decoded

        llm2vec = getattr(self.text_encoder, "model", None)
        tokenizer = getattr(llm2vec, "tokenizer", None)
        llm_model = getattr(llm2vec, "model", None)
        if tokenizer is None or llm_model is None or not hasattr(llm_model, "generate"):
            raise RuntimeError("Shared LLM generation is unavailable for this text encoder")

        max_length = int(getattr(llm2vec, "max_length", 512))
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_len = int(encoded["input_ids"].shape[1]) if "input_ids" in encoded else 0

        device = self.text_encoder.get_device() if hasattr(self.text_encoder, "get_device") else "cpu"
        encoded = {key: value.to(device) for key, value in encoded.items()}

        do_sample = float(temperature) > 1e-6
        generate_kwargs = {
            "max_new_tokens": max(8, int(max_new_tokens)),
            "do_sample": do_sample,
            "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = int(eos_token_id)
        if do_sample:
            generate_kwargs["temperature"] = float(max(temperature, 1e-6))
            generate_kwargs["top_p"] = float(min(1.0, max(0.1, top_p)))

        with torch.no_grad():
            output = llm_model.generate(**encoded, **generate_kwargs)

        generated_ids = output[0, input_len:] if input_len > 0 and output.shape[1] > input_len else output[0]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not decoded:
            full_decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            if full_decoded.lower().startswith(prompt.lower()):
                full_decoded = full_decoded[len(prompt) :].strip()
            decoded = full_decoded
        if not decoded:
            raise RuntimeError("Shared LLM generation returned empty text")
        return decoded

    def __call__(self, text, filename, progress=gr.Progress()):
        # Compute text embedding
        tensor, length = self.text_encoder(text)
        embedding = tensor[:length]
        embedding = embedding.cpu().numpy()

        # Save text embedding
        path = os.path.join(self.tmp_folder, filename)
        np.save(path, embedding)

        output_title = gr.Markdown(visible=True)
        output_text = gr.Markdown(visible=True, value=f"Text: {text}")
        download = gr.DownloadButton(visible=True, value=path)
        return download, output_title, output_text

    def generate_description(self, prompt, max_new_tokens=96, temperature=0.2, top_p=0.9):
        prompt = str(prompt).strip()
        if not prompt:
            return ""
        generated = self._generate_with_shared_llm(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        return generated.strip()


def _get_env(name: str, default):
    return os.getenv(name, default)


def _build_text_encoder(name: str, fp32: bool = False):
    if name not in TEXT_ENCODER_PRESETS:
        available = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unknown TEXT_ENCODER='{name}'. Available: {available}")
    preset = TEXT_ENCODER_PRESETS[name]
    target_cls = resolve_target(preset["target"])
    if fp32:
        preset["kwargs"]["dtype"] = "float32"
    return target_cls(**preset["kwargs"])


def parse_args():
    parser = argparse.ArgumentParser(description="Run text encoder Gradio server.")
    parser.add_argument(
        "--text-encoder",
        default=_get_env("TEXT_ENCODER", DEFAULT_TEXT_ENCODER),
        choices=sorted(TEXT_ENCODER_PRESETS.keys()),
        help="Text encoder preset.",
    )
    parser.add_argument(
        "--tmp-folder",
        default=_get_env("TEXT_ENCODER_TMP_FOLDER", DEFAULT_TMP_FOLDER),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Uses fp32 for the text encoder rather than default bfloat16.",
    )
    parser.add_argument(
        "--enable-generation-model",
        action="store_true",
        default=DEFAULT_ENABLE_GENERATION_MODEL,
        help="Load a separate causal LM for /generate_description endpoint",
    )
    parser.add_argument(
        "--generation-model-name-or-path",
        default=_get_env("GENERATION_MODEL_NAME_OR_PATH", DEFAULT_GENERATION_MODEL),
        help="Causal LM model name/path for generation endpoint",
    )
    parser.add_argument(
        "--generation-quantization",
        choices=["none", "4bit", "8bit"],
        default=_get_env("GENERATION_QUANTIZATION", DEFAULT_GENERATION_QUANTIZATION),
        help="Quantization mode for generation model",
    )
    return parser.parse_args()


def _build_generation_model(name_or_path: str, quantization: str):
    quant = str(quantization).strip().lower()
    model_kwargs = {"device_map": "auto"}
    if quant in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                f"Requested generation quantization='{quant}' but BitsAndBytesConfig is unavailable: {exc}"
            ) from exc
        if quant == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    server_name = _get_env("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME)
    server_port = int(_get_env("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT))
    theme, css = get_gradio_theme()
    tmp_folder = os.path.abspath(os.path.expanduser(args.tmp_folder))
    os.makedirs(tmp_folder, exist_ok=True)
    text_encoder = _build_text_encoder(args.text_encoder, args.fp32)
    generation_tokenizer = None
    generation_model = None
    if bool(args.enable_generation_model):
        generation_tokenizer, generation_model = _build_generation_model(
            str(args.generation_model_name_or_path),
            str(args.generation_quantization),
        )
    display_name = TEXT_ENCODER_PRESETS[args.text_encoder]["display_name"]
    demo_wrapper_fn = DemoWrapper(
        text_encoder,
        tmp_folder,
        generation_tokenizer=generation_tokenizer,
        generation_model=generation_model,
    )

    with gr.Blocks(title="Text encoder") as demo:
        gr.Markdown(f"# Text encoder: {display_name}")
        gr.Markdown("## Description")
        gr.Markdown("Get a embeddings from a text.")

        gr.Markdown("## Inputs")
        with gr.Row():
            text = gr.Textbox(
                placeholder="Type the motion you want to generate with a sentence",
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
                type="text",
            )
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                btn = gr.Button("Encode", variant="primary")
            with gr.Column(scale=1):
                clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=3):
                pass

        output_title = gr.Markdown("## Outputs", visible=False)
        output_text = gr.Markdown("", visible=False)
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                download = gr.DownloadButton("Download", variant="primary", visible=False)
            with gr.Column(scale=4):
                pass

        filename = gr.Textbox(
            visible=False,
            value="embedding.npy",
        )
        gen_prompt = gr.Textbox(visible=False, value="")
        gen_max_new_tokens = gr.Number(visible=False, value=96, precision=0)
        gen_temperature = gr.Number(visible=False, value=0.2)
        gen_top_p = gr.Number(visible=False, value=0.9)
        gen_button = gr.Button(visible=False)
        gen_output = gr.Textbox(visible=False, value="")

        def clear_fn():
            return [
                gr.DownloadButton(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
            ]

        outputs = [download, output_title, output_text]

        gr.on(
            triggers=[text.submit, btn.click],
            fn=clear_fn,
            inputs=None,
            outputs=outputs,
        ).then(
            fn=demo_wrapper_fn,
            inputs=[text, filename],
            outputs=outputs,
        )

        def download_file():
            return gr.DownloadButton()

        download.click(
            fn=download_file,
            inputs=None,
            outputs=[download],
        )
        clear.click(fn=clear_fn, inputs=None, outputs=outputs)
        gen_button.click(
            fn=demo_wrapper_fn.generate_description,
            inputs=[gen_prompt, gen_max_new_tokens, gen_temperature, gen_top_p],
            outputs=[gen_output],
            api_name="generate_description",
        )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        theme=theme,
        css=css,
        allowed_paths=[tmp_folder],
        show_error=True,
    )


if __name__ == "__main__":
    main()
