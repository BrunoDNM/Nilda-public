"""Avalia somente NewCare e MiniCare INT8 original preservado."""

import argparse
from collections import Counter
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import binomtest


BASE = Path(__file__).resolve().parent
NEW_DATA = BASE / "newcare/model/dados"
MINI_DATA = BASE / "minicare/model/dados"
THRESHOLD = 0.5


def score_rates(sensitivity, specificity):
    precision_denominator = sensitivity + (1 - specificity)
    return {
        "accuracy": (sensitivity + specificity) / 2,
        "precision": sensitivity / precision_denominator if precision_denominator else None,
        "recall": sensitivity,
        "f1": 2 * sensitivity / (2 + sensitivity - specificity),
    }


def summarize_predictions(labels, probabilities):
    """Calcula estimativas e ICs simultâneos de 95% por modelo (desenho balanceado)."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Rótulos e probabilidades devem ser vetores de mesmo tamanho.")
    if (
        not np.isin(labels, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Esperados rótulos binários e probabilidades no intervalo [0, 1].")
    positive = labels == 1
    negative = ~positive
    n = int(np.count_nonzero(positive))
    if not n or np.count_nonzero(negative) != n:
        raise ValueError("Este IC pressupõe número igual e não nulo de sessões por classe.")

    predicted = probabilities > THRESHOLD
    tp = int(np.count_nonzero(positive & predicted))
    tn = int(np.count_nonzero(negative & ~predicted))
    sensitivity_ci = binomtest(tp, n).proportion_ci(0.975, method="exact")
    specificity_ci = binomtest(tn, n).proportion_ci(0.975, method="exact")
    points = score_rates(tp / n, tn / n)
    lower = score_rates(sensitivity_ci.low, specificity_ci.low)
    upper = score_rates(sensitivity_ci.high, specificity_ci.high)
    confidence = np.where(predicted, probabilities, 1 - probabilities)
    errors = predicted != positive

    return {
        "confusion": {"tp": tp, "fn": n - tp, "fp": n - tn, "tn": tn},
        "metrics": {
            name: {"estimate": value, "ci95": [lower[name], upper[name]]}
            for name, value in points.items()
        },
        "confidence": {
            "true_positives_mean": float(confidence[positive & predicted].mean()) if tp else None,
            "errors_mean": float(confidence[errors].mean()) if errors.any() else None,
        },
    }


def load_sessions(records):
    sessions = {}
    for entry in records:
        label = entry["stress"]
        if not isinstance(label, (bool, int)) or label not in (0, 1):
            raise ValueError("Rótulo stress inválido.")
        points, session_label = sessions.setdefault(entry["date"], ([], int(label)))
        if label != session_label:
            raise ValueError("Uma sessão possui rótulos conflitantes.")
        points.append((entry["x"], entry["y"]))

    trajectories, labels = [], []
    for points, label in (sessions[key] for key in sorted(sessions)):
        trajectory = np.asarray(points, dtype=float)
        if len(trajectory) < 2 or not np.isfinite(trajectory).all():
            raise ValueError("Uma sessão possui pontos insuficientes ou coordenadas não finitas.")
        trajectories.append(trajectory)
        labels.append(label)
    if Counter(labels) != {0: 10, 1: 10}:
        raise ValueError("Esperadas as 20 sessões da Tabela 2, com 10 por classe.")
    return trajectories, labels


def describe_tensor(detail):
    scale, zero = detail["quantization"]
    return {
        "shape": detail["shape"].tolist(),
        "dtype": np.dtype(detail["dtype"]).name,
        "scale": scale,
        "zero_point": zero,
    }


def infer_tflite(interpreter, images, features, *, legacy_uint8_cast=False):
    interpreter.allocate_tensors()
    inputs = {tuple(detail["shape"]): detail for detail in interpreter.get_input_details()}
    if inputs.keys() != {(1, 64, 64, 1), (1, 4)}:
        raise ValueError("Esperadas entradas TFLite de imagem 64 × 64 e quatro características.")
    image_input = inputs[(1, 64, 64, 1)]
    feature_input = inputs[(1, 4)]
    output, = interpreter.get_output_details()
    if tuple(output["shape"]) != (1, 1):
        raise ValueError("Esperada uma probabilidade por sessão na saída TFLite.")
    integer = np.issubdtype(output["dtype"], np.integer)
    for detail in (image_input, feature_input, output):
        scale, _ = detail["quantization"]
        dtype = detail["dtype"]
        if integer and (dtype not in (np.int8, np.uint8) or scale <= 0):
            raise ValueError("Esperados tensores de 8 bits com quantização por tensor.")
        if not integer and dtype != np.float32:
            raise ValueError("Esperados tensores float32 para o modelo TFLite de ponto flutuante.")
        if legacy_uint8_cast and dtype != np.uint8:
            raise ValueError("O caminho histórico requer tensores uint8.")

    predictions = []
    for image, feature in zip(images, features, strict=True):
        for detail, value in ((image_input, image[None]), (feature_input, feature[None])):
            dtype = detail["dtype"]
            if integer:
                scale, zero = detail["quantization"]
                value = value / scale
                if legacy_uint8_cast:
                    value = value + zero  # O conversor original truncava no cast, sem saturar.
                else:
                    rounded = np.copysign(np.floor(np.abs(value) + 0.5), value)
                    bounds = np.iinfo(dtype)
                    value = np.clip(rounded + zero, bounds.min, bounds.max)
            interpreter.set_tensor(detail["index"], value.astype(dtype))
        interpreter.invoke()
        value = float(interpreter.get_tensor(output["index"]).item())
        if integer:
            scale, zero = output["quantization"]
            value = (value - zero) * scale
        predictions.append(value)

    return predictions, {
        "image": describe_tensor(image_input),
        "features": describe_tensor(feature_input),
        "output": describe_tensor(output),
    }


def evaluate(args):
    import joblib
    import tensorflow as tf

    sys.path.insert(0, str(BASE / "newcare"))
    from newcare.Config import Config
    from newcare.trajectory import (
        extract_features_from_trajectory,
        load_json_data,
        render_trajectory,
    )

    paths = {
        "evaluation_data": args.evaluation_data,
        "feature_scaler": args.scaler,
        "newcare_h5": args.newcare_model,
        "minicare_tflite_int8": args.minicare_int8,
    }
    inputs = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            inputs[name] = {"filename": path.name, "sha256": file_digest(handle, "sha256").hexdigest()}

    trajectories, labels = load_sessions(load_json_data(args.evaluation_data))
    scaler = joblib.load(args.scaler)
    features = scaler.transform(np.array([
        extract_features_from_trajectory(points) for points in trajectories
    ])).astype(np.float32)

    def images(size):
        return np.concatenate([
            render_trajectory(
                points,
                img_height=Config("TRAJECTORY_HEIGHT"),
                img_width=Config("TRAJECTORY_WIDTH"),
                target_size=size,
            )
            for points in trajectories
        ])

    new_images = images(128)
    mini_images = images(64)

    def load_model(path):
        return tf.keras.models.load_model(path, compile=False, safe_mode=True)

    def keras_scores(model, inputs):
        return model(inputs, training=False).numpy().reshape(-1).tolist()

    new_model = load_model(args.newcare_model)
    scores = {
        "newcare_h5": keras_scores(new_model, [new_images, features]),
    }
    # Apenas o MiniCare INT8 preservado integra a comparação científica.
    interpreter = tf.lite.Interpreter(model_path=str(paths["minicare_tflite_int8"]), num_threads=1)
    scores["minicare_tflite_int8"], int8_contract = infer_tflite(
        interpreter, mini_images, features
    )
    tensor_details = {"minicare_tflite_int8": int8_contract}

    report = {
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "packages": {name: version(name) for name in (
                "tensorflow-cpu", "keras", "numpy", "scipy", "scikit-learn", "scikit-image",
                "joblib", "pillow",
            )},
        },
        "inputs": inputs,
        "session_counts": {"positive": 10, "negative": 10},
        "positive_if_strictly_greater_than": THRESHOLD,
        "interval_method": "Clopper-Pearson 97.5% por taxa e projeção Bonferroni, 95% por modelo",
        "model_parameters": {
            "newcare_h5": new_model.count_params(),
        },
        "tflite_tensors": tensor_details,
        "results": {name: summarize_predictions(labels, values) for name, values in scores.items()},
    }
    predictions = [
        {"session": index, "label": label, "probabilities": {name: p[index - 1] for name, p in scores.items()}}
        for index, label in enumerate(labels, start=1)
    ]
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-data", type=Path, default=NEW_DATA / "dataset-validacao.json")
    parser.add_argument("--scaler", type=Path, default=NEW_DATA / "feature_scaler.save")
    parser.add_argument("--newcare-model", type=Path, default=NEW_DATA / "modelCNN-multi-1784648730.h5")
    parser.add_argument("--minicare-int8", type=Path, default=MINI_DATA / "modelCNN-minicare-int8.tflite")
    parser.add_argument("--output-dir", type=Path, help="Novo diretório para métricas e predições.")
    args = parser.parse_args()
    if args.output_dir and args.output_dir.exists():
        parser.error("--output-dir deve indicar um diretório ainda inexistente.")

    report, predictions = evaluate(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output_dir:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "evaluation.json").write_text(serialized, encoding="utf-8")
        (args.output_dir / "predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(serialized, end="")


if __name__ == "__main__":
    main()
