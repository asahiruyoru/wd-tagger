import argparse
import ast
import csv
import json
from pathlib import Path

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

TITLE = "WaifuDiffusion / PixAI Tagger"
DESCRIPTION = """
Demo for the WaifuDiffusion tagger models and the newer PixAI tagger v0.9.

Model files are loaded from local subdirectories under `./models/`. If a
selected model is missing locally, it is downloaded from Hugging Face on
first use and cached into `./models/<dir>/`.
"""


# --- Model registry -----------------------------------------------------------
# Each entry maps a display label to:
#   - dir:     subdirectory under MODELS_ROOT that holds the model files
#   - kind:    preprocessing / label-loading pipeline
#   - hf_repo: Hugging Face repo to download model.onnx + selected_tags.csv from
#              when the local files are missing
#
# kind = "wd"     : WD/IdolSankaku family (NHWC, BGR, pad-to-square, sigmoid-applied
#                   output, tag categories 0=general / 4=character / 9=rating)
# kind = "pixai"  : PixAI v0.9 family     (NCHW, RGB, stretch-resize, raw logits
#                   -> sigmoid, tag categories 0=general / 4=character / + `ips`
#                   column; no rating)
MODELS_ROOT = Path(__file__).resolve().parent / "models"

MODEL_REGISTRY = {
    # --- PixAI (newest) ---
    "pixai-tagger-v0.9 (ONNX)":          {"dir": "pixai-tagger-v0.9-onnx",            "kind": "pixai", "hf_repo": "deepghs/pixai-tagger-v0.9-onnx"},
    # --- WD v3 ---
    "wd-eva02-large-tagger-v3":          {"dir": "wd-eva02-large-tagger-v3",          "kind": "wd",    "hf_repo": "SmilingWolf/wd-eva02-large-tagger-v3"},
    "wd-vit-large-tagger-v3":            {"dir": "wd-vit-large-tagger-v3",            "kind": "wd",    "hf_repo": "SmilingWolf/wd-vit-large-tagger-v3"},
    "wd-swinv2-tagger-v3":               {"dir": "wd-swinv2-tagger-v3",               "kind": "wd",    "hf_repo": "SmilingWolf/wd-swinv2-tagger-v3"},
    "wd-convnext-tagger-v3":             {"dir": "wd-convnext-tagger-v3",             "kind": "wd",    "hf_repo": "SmilingWolf/wd-convnext-tagger-v3"},
    "wd-vit-tagger-v3":                  {"dir": "wd-vit-tagger-v3",                  "kind": "wd",    "hf_repo": "SmilingWolf/wd-vit-tagger-v3"},
    # --- WD v2 ---
    "wd-v1-4-moat-tagger-v2":            {"dir": "wd-v1-4-moat-tagger-v2",            "kind": "wd",    "hf_repo": "SmilingWolf/wd-v1-4-moat-tagger-v2"},
    "wd-v1-4-swinv2-tagger-v2":          {"dir": "wd-v1-4-swinv2-tagger-v2",          "kind": "wd",    "hf_repo": "SmilingWolf/wd-v1-4-swinv2-tagger-v2"},
    "wd-v1-4-convnext-tagger-v2":        {"dir": "wd-v1-4-convnext-tagger-v2",        "kind": "wd",    "hf_repo": "SmilingWolf/wd-v1-4-convnext-tagger-v2"},
    "wd-v1-4-convnextv2-tagger-v2":      {"dir": "wd-v1-4-convnextv2-tagger-v2",      "kind": "wd",    "hf_repo": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"},
    "wd-v1-4-vit-tagger-v2":             {"dir": "wd-v1-4-vit-tagger-v2",             "kind": "wd",    "hf_repo": "SmilingWolf/wd-v1-4-vit-tagger-v2"},
    # --- IdolSankaku ---
    "idolsankaku-swinv2-tagger-v1":      {"dir": "idolsankaku-swinv2-tagger-v1",      "kind": "wd",    "hf_repo": "deepghs/idolsankaku-swinv2-tagger-v1"},
    "idolsankaku-eva02-large-tagger-v1": {"dir": "idolsankaku-eva02-large-tagger-v1", "kind": "wd",    "hf_repo": "deepghs/idolsankaku-eva02-large-tagger-v1"},
}

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()


def ensure_model_files(model_label):
    """Return (csv_path, model_path, kind), downloading from HF on first use.

    Files land in `MODELS_ROOT/<dir>/` so subsequent runs reuse them without
    touching the network.
    """
    info = MODEL_REGISTRY[model_label]
    model_dir = MODELS_ROOT / info["dir"]
    model_path = model_dir / MODEL_FILENAME
    csv_path = model_dir / LABEL_FILENAME

    if not (model_path.is_file() and csv_path.is_file()):
        repo_id = info.get("hf_repo")
        if not repo_id:
            raise FileNotFoundError(
                f"Local files for '{model_label}' are missing and no Hugging Face "
                f"repo is registered. Expected:\n  {model_path}\n  {csv_path}"
            )
        print(f"[wd-tagger] downloading '{model_label}' from {repo_id} -> {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(repo_id, MODEL_FILENAME, local_dir=str(model_dir))
        huggingface_hub.hf_hub_download(repo_id, LABEL_FILENAME, local_dir=str(model_dir))

    return str(csv_path), str(model_path), info["kind"]


def load_labels_wd(dataframe):
    """Load labels for WD-format models.

    WD models use categories: 0=general, 4=character, 9=rating.
    """
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def load_labels_pixai(csv_path):
    """Load labels for PixAI-format models.

    PixAI's selected_tags.csv has columns: id, name, category, ips.
    Category 4 = character, others = general. No rating category.
    The `ips` column maps a character tag to a list of copyright/IP tags.

    The `id` column is used DIRECTLY as the position in the model's output
    vector. The ONNX output has length = (max id + 1), so we build a
    tag_names list of that length, with empty strings at any gaps.
    """
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            name = row.get("name", "")
            try:
                category = int(row.get("category", 0))
            except (TypeError, ValueError):
                category = 0

            ips_raw = row.get("ips", "[]") or "[]"
            ips = []
            if ips_raw and ips_raw != "[]":
                try:
                    ips = json.loads(ips_raw)
                except Exception:
                    try:
                        ips = ast.literal_eval(ips_raw)
                    except Exception:
                        ips = []

            entries.append(
                {
                    "id": idx,
                    "name": name,
                    "is_char": category == 4,
                    "ips": [str(ip) for ip in ips],
                }
            )

    if not entries:
        return [], [], [], {}

    max_id = max(item["id"] for item in entries)
    size = max_id + 1

    tag_names = [""] * size
    general_indexes = []
    character_indexes = []
    ips_mapping = {}

    for item in entries:
        i = item["id"]
        display_name = item["name"]
        if display_name not in kaomojis:
            display_name = display_name.replace("_", " ")
        tag_names[i] = display_name

        if item["is_char"]:
            character_indexes.append(i)
            if item["ips"]:
                ips_mapping[display_name] = [
                    ip.replace("_", " ") if ip not in kaomojis else ip
                    for ip in item["ips"]
                ]
        else:
            general_indexes.append(i)

    return tag_names, general_indexes, character_indexes, ips_mapping


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_label = None
        self.kind = None
        self.ips_mapping = {}
        self.output_name = None

    def load_model(self, model_label):
        if model_label == self.last_loaded_label:
            return

        csv_path, model_path, kind = ensure_model_files(model_label)
        self.kind = kind

        if kind == "pixai":
            tag_names, general_idx, character_idx, ips_mapping = load_labels_pixai(
                csv_path
            )
            self.tag_names = tag_names
            self.rating_indexes = []  # PixAI has no rating category
            self.general_indexes = general_idx
            self.character_indexes = character_idx
            self.ips_mapping = ips_mapping
        else:
            tags_df = pd.read_csv(csv_path)
            sep_tags = load_labels_wd(tags_df)
            self.tag_names = sep_tags[0]
            self.rating_indexes = sep_tags[1]
            self.general_indexes = sep_tags[2]
            self.character_indexes = sep_tags[3]
            self.ips_mapping = {}

        model = rt.InferenceSession(model_path)

        input_shape = model.get_inputs()[0].shape
        if kind == "pixai":
            # PixAI is NCHW. Prefer the model's declared spatial size; default to 448.
            self.model_target_size = 448
            try:
                h = input_shape[2]
                if isinstance(h, int) and h > 0:
                    self.model_target_size = h
            except Exception:
                pass
        else:
            # WD is NHWC: (N, H, W, C)
            _, height, _, _ = input_shape
            self.model_target_size = height

        self.output_name = self.select_output_name(model, len(self.tag_names))
        self.last_loaded_label = model_label
        self.model = model

    @staticmethod
    def select_output_name(model, expected_dim):
        """Pick the output whose class dimension matches the tag count.

        PixAI's exported ONNX can expose multiple outputs such as embeddings and
        tag logits. Reading the first output blindly can map an embedding vector
        onto tag names, which produces nonsensical high-confidence labels.
        """
        outputs = model.get_outputs()
        if not outputs:
            raise RuntimeError("Model has no outputs.")

        fallback_name = outputs[0].name
        fallback_dim = -1

        for output in outputs:
            shape = output.shape or []
            numeric_dims = [dim for dim in shape if isinstance(dim, int) and dim > 0]
            if expected_dim in numeric_dims:
                return output.name

            if numeric_dims:
                candidate_dim = max(numeric_dims)
                if candidate_dim > fallback_dim:
                    fallback_name = output.name
                    fallback_dim = candidate_dim

        return fallback_name

    @staticmethod
    def ensure_probabilities(values):
        """Convert logits to probabilities, but don't sigmoid real probabilities twice."""
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if 0.0 <= min_value and max_value <= 1.0:
            return values

        clipped = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def prepare_image_wd(self, image):
        """WD preprocessing: pad to square (white bg), resize, RGB->BGR, NHWC, float32 0-255."""
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # RGB -> BGR

        return np.expand_dims(image_array, axis=0)

    def prepare_image_pixai(self, image):
        """PixAI preprocessing: direct resize to 448x448, RGB, normalize to [-1, 1], NCHW.

        Note: this does NOT pad to square — it stretches. This matches the
        reference implementation from deepghs / DraconicDragon.
        """
        target_size = self.model_target_size

        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((target_size, target_size), Image.BICUBIC)
        img = np.asarray(image, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5  # -> [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0).astype(np.float32)

    def predict(
        self,
        image,
        model_label,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
        resolve_ip_mapping,
    ):
        self.load_model(model_label)

        if self.kind == "pixai":
            input_tensor = self.prepare_image_pixai(image)
        else:
            input_tensor = self.prepare_image_wd(image)

        input_name = self.model.get_inputs()[0].name
        preds = self.model.run([self.output_name], {input_name: input_tensor})[0]

        # PixAI's ONNX output is raw logits; WD output is already sigmoid-applied probabilities.
        if self.kind == "pixai":
            preds = self.ensure_probabilities(preds)

        probs = np.asarray(preds, dtype=np.float32)
        if probs.ndim > 1:
            probs = probs[0]
        probs = probs.reshape(-1).astype(float)
        n = min(len(self.tag_names), len(probs))
        labels = list(zip(self.tag_names[:n], probs[:n]))

        def _gather(indexes):
            # Skip any index that is out of range for the current model output.
            return [labels[i] for i in indexes if i < n]

        # Ratings (WD only)
        if self.rating_indexes:
            ratings_names = _gather(self.rating_indexes)
            rating = dict(ratings_names)
        else:
            rating = {}

        # General tags
        general_names = _gather(self.general_indexes)
        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)
        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Character tags
        character_names = _gather(self.character_indexes)
        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)
        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        # Copyright / IP mapping (PixAI only)
        ip_string = ""
        if self.kind == "pixai" and resolve_ip_mapping and self.ips_mapping:
            detected_ips = set()
            for char_name in character_res.keys():
                if char_name in self.ips_mapping:
                    for ip in self.ips_mapping[char_name]:
                        detected_ips.add(ip)
            if detected_ips:
                ip_string = ", ".join(sorted(detected_ips))

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", r"\(").replace(")", r"\)")
        )

        return sorted_general_strings, rating, character_res, general_res, ip_string


def main():
    args = parse_args()

    predictor = Predictor()
    dropdown_list = list(MODEL_REGISTRY.keys())
    default_choice = dropdown_list[0]

    with gr.Blocks(title=TITLE) as demo:
        with gr.Column():
            gr.Markdown(
                value=f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
            )
            gr.Markdown(value=DESCRIPTION)
            with gr.Row():
                with gr.Column(variant="panel"):
                    image = gr.Image(type="pil", image_mode="RGBA", label="Input")
                    model_repo = gr.Dropdown(
                        dropdown_list,
                        value=default_choice,
                        label="Model (downloaded on first use into ./models/)",
                    )
                    with gr.Row():
                        general_thresh = gr.Slider(
                            0,
                            1,
                            step=args.score_slider_step,
                            value=args.score_general_threshold,
                            label="General Tags Threshold",
                            scale=3,
                        )
                        general_mcut_enabled = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold",
                            scale=1,
                        )
                    with gr.Row():
                        character_thresh = gr.Slider(
                            0,
                            1,
                            step=args.score_slider_step,
                            value=args.score_character_threshold,
                            label="Character Tags Threshold",
                            scale=3,
                        )
                        character_mcut_enabled = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold",
                            scale=1,
                        )
                    resolve_ip_mapping = gr.Checkbox(
                        value=True,
                        label="Resolve Copyright/IP mapping (PixAI only)",
                    )
                    with gr.Row():
                        clear = gr.ClearButton(
                            components=[
                                image,
                                model_repo,
                                general_thresh,
                                general_mcut_enabled,
                                character_thresh,
                                character_mcut_enabled,
                                resolve_ip_mapping,
                            ],
                            variant="secondary",
                            size="lg",
                        )
                        submit = gr.Button(value="Submit", variant="primary", size="lg")
                with gr.Column(variant="panel"):
                    sorted_general_strings = gr.Textbox(label="Output (string)")
                    rating = gr.Label(label="Rating (WD only)")
                    character_res = gr.Label(label="Output (characters)")
                    ip_output = gr.Textbox(label="Copyright / IP (PixAI only)")
                    general_res = gr.Label(label="Output (tags)")
                    clear.add(
                        [
                            sorted_general_strings,
                            rating,
                            character_res,
                            ip_output,
                            general_res,
                        ]
                    )

        submit.click(
            predictor.predict,
            inputs=[
                image,
                model_repo,
                general_thresh,
                general_mcut_enabled,
                character_thresh,
                character_mcut_enabled,
                resolve_ip_mapping,
            ],
            outputs=[
                sorted_general_strings,
                rating,
                character_res,
                general_res,
                ip_output,
            ],
        )

    demo.queue(max_size=10)
    demo.launch()


if __name__ == "__main__":
    main()
