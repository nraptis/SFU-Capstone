from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Final

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from ml.classify import classify
from ml.pretty_print import to_pretty_print
from ml.preprocess import preprocess
from ui.data import CELL_TYPE_CARDS

ALLOWED_EXTENSIONS: Final[set[str]] = {"png", "jpg", "jpeg"}
MAX_UPLOAD_MB: Final[int] = 5
IMAGE_DIRECTORY: Final[Path] = Path(__file__).resolve().parent / "ui" / "images"

app = Flask(__name__)
application = app


def _is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def _read_uploaded_image(uploaded_file) -> Image.Image:
    raw_bytes = uploaded_file.read()
    size_bytes = len(raw_bytes)
    size_mb = float(size_bytes) / (1024.0 * 1024.0)
    if size_mb > float(MAX_UPLOAD_MB):
        raise ValueError(f"File too large ({size_mb:.1f} MB). Max is {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _pil_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _cell_type_images() -> list[str]:
    return [
        "basophil_grid.jpg",
        "eosinophil_grid.jpg",
        "hairy_cell_grid.jpg",
        "lymphocyte_grid.jpg",
        "lymphocyte_large_granular_grid.jpg",
        "lymphocyte_neoplastic_grid.jpg",
        "metamyelocyte_grid.jpg",
        "monocyte_grid.jpg",
        "myeloblast_grid.jpg",
        "myelocyte_grid.jpg",
        "neutrophil_band_grid.jpg",
        "neutrophil_segmented_grid.jpg",
        "plasma_cell_grid.jpg",
        "promyelocyte_grid.jpg",
        "promyelocyte_atypical_grid.jpg",
        "normoblast_grid.jpg",
    ]


def _cell_type_cards_for_grid() -> list[dict[str, str]]:
    title_to_card = {card.title: card for card in CELL_TYPE_CARDS}
    filename_to_title = {
        "basophil_grid.jpg": "Basophil",
        "eosinophil_grid.jpg": "Eosinophil",
        "hairy_cell_grid.jpg": "Hairy Cell",
        "lymphocyte_grid.jpg": "Lymphocyte",
        "lymphocyte_large_granular_grid.jpg": "Lymphocyte (Large Granular)",
        "lymphocyte_neoplastic_grid.jpg": "Lymphocyte (Neoplastic)",
        "metamyelocyte_grid.jpg": "Metamyelocyte",
        "monocyte_grid.jpg": "Monocyte",
        "myeloblast_grid.jpg": "Myeloblast",
        "myelocyte_grid.jpg": "Myelocyte",
        "neutrophil_band_grid.jpg": "Neutrophil Band",
        "neutrophil_segmented_grid.jpg": "Neutrophil Segmented",
        "plasma_cell_grid.jpg": "Plasma Cell",
        "promyelocyte_grid.jpg": "Promyelocyte",
        "promyelocyte_atypical_grid.jpg": "Promyelocyte (Atypical)",
        "normoblast_grid.jpg": "Normoblast",
    }
    items: list[dict[str, str]] = []
    for filename in _cell_type_images():
        title = filename_to_title.get(filename, filename)
        card = title_to_card.get(title)
        items.append(
            {
                "image_name": filename,
                "title": title,
                "summary": card.summary if card else "",
                "concern": card.concern if card else "",
            }
        )
    return items


def _preview_caption(normalize: bool) -> str:
    if normalize:
        return "Model input preview: center-cropped 224x224 with LAB normalization."
    return "Model input preview: center-cropped 224x224 without LAB normalization."


@app.get("/images/<path:filename>")
def local_image(filename: str):
    if not IMAGE_DIRECTORY.exists():
        abort(404)
    return send_from_directory(IMAGE_DIRECTORY, filename)


@app.post("/preview-model-input")
def preview_model_input():
    uploaded_file = request.files.get("cell_image")
    normalize = request.form.get("normalize") == "on"

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400
    if not _is_allowed_file(uploaded_file.filename):
        return jsonify({"error": "Allowed file types: PNG, JPG, JPEG."}), 400

    try:
        image = _read_uploaded_image(uploaded_file)
        model_input = preprocess(image=image, normalize=bool(normalize))
        return jsonify(
            {
                "preview_uri": _pil_to_data_uri(model_input),
                "preview_caption": _preview_caption(normalize=bool(normalize)),
                "selected_filename": uploaded_file.filename,
                "original_size": {"width": image.width, "height": image.height},
                "cropped_size": {"width": model_input.width, "height": model_input.height},
                "lab_normalized": bool(normalize),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/", methods=["GET", "POST"])
def index():
    selected_filename = "No file selected"
    normalize = True
    active_tab = "blood-test"
    message = ""
    error = ""
    preview_uri = ""
    preview_caption = ""
    classified_preview_uri_with_norm = ""
    classified_preview_uri_without_norm = ""
    image_stats_rows: list[dict[str, str]] = []
    ensemble_rows: list[dict[str, str]] = []
    model_rows: dict[str, list[str]] = {}

    if request.method == "POST":
        active_tab = request.form.get("active_tab", "blood-test")
        uploaded_file = request.files.get("cell_image")
        normalize = request.form.get("normalize") == "on"

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an image file."
        elif not _is_allowed_file(uploaded_file.filename):
            error = "Allowed file types: PNG, JPG, JPEG."
        else:
            selected_filename = uploaded_file.filename
            try:
                image = _read_uploaded_image(uploaded_file)
                model_input = preprocess(image=image, normalize=bool(normalize))
                preview_uri = _pil_to_data_uri(model_input)
                preview_caption = _preview_caption(normalize=bool(normalize))
                classified_preview_uri_with_norm = _pil_to_data_uri(
                    preprocess(image=image, normalize=True)
                )
                classified_preview_uri_without_norm = _pil_to_data_uri(
                    preprocess(image=image, normalize=False)
                )

                result = classify(image=image, normalize=bool(normalize))
                pretty_result = to_pretty_print(result=result, top_k_ensemble=8)
                image_stats_rows = [
                    {"stat_name": "Original Size", "stat_value": f"{image.width} x {image.height}"},
                    {
                        "stat_name": "Cropped Size",
                        "stat_value": f"{model_input.width} x {model_input.height}",
                    },
                    {
                        "stat_name": "Lab Normalized",
                        "stat_value": "True" if normalize else "False",
                    },
                ]

                ensemble_rows = [
                    {
                        "class_name": item.class_name,
                        "probability": f"{item.probability * 100.0:.2f}%",
                    }
                    for item in pretty_result.ensemble_list
                ]

                model_rows = {
                    model_name: [
                        f"{class_name}: {pretty_result.model_result_table[model_name].class_table[class_name] * 100.0:.2f}%"
                        for class_name in pretty_result.model_result_table[model_name].class_list
                    ]
                    for model_name in pretty_result.model_list
                }
                message = "Inference complete. Upload a new image to run analysis again"
            except Exception as exc:
                error = f"Inference failed: {exc}"

    return render_template(
        "index.html",
        banner_exists=(IMAGE_DIRECTORY / "banner.jpg").exists(),
        cell_type_cards=_cell_type_cards_for_grid(),
        active_tab=active_tab,
        selected_filename=selected_filename,
        normalize=normalize,
        message=message,
        error=error,
        preview_uri=preview_uri,
        preview_caption=preview_caption,
        classified_preview_uri_with_norm=classified_preview_uri_with_norm,
        classified_preview_uri_without_norm=classified_preview_uri_without_norm,
        image_stats_rows=image_stats_rows,
        ensemble_rows=ensemble_rows,
        model_rows=model_rows,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
