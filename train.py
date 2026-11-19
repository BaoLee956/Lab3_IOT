"""Train, test once, export a single-file YOLO11n ONNX and check its contract."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from ultralytics import YOLO

NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def record_metrics(metrics, model, split, target):
    per_class = []
    for i, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class.append({"class_id": int(cls), "name": NAMES[int(cls)],
                          "precision": float(p), "recall": float(r), "f1": float(f1),
                          "AP50": float(ap50), "AP50_95": float(ap)})
    payload = {"model": str(model), "split": split, "per_class": per_class,
               "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
               "speed_ms_per_image": metrics.speed,
               "note": "Ultralytics validation metrics; not browser-to-cloud latency."}
    Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for CUDA")
    args = parser.parse_args()
    evidence = Path("evidence")
    evidence.mkdir(exist_ok=True)
    # Override the path to an absolute path at runtime; do not rewrite the source YAML.
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    config["path"] = str(Path(args.data).resolve().parent)
    expected = [config["names"][i] for i in range(3)]
    if expected != NAMES:
        raise ValueError("Incorrect class mapping in dataset YAML")
    runtime_yaml = evidence / "data-runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = YOLO("yolo11n.pt")
    model.train(data=str(runtime_yaml.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, seed=42,
                deterministic=True, patience=12, cache=False,
                project="runs", name="mask_train", exist_ok=False)
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    test = trained.val(data=str(runtime_yaml.resolve()), split="test", imgsz=args.imgsz,
                       batch=1, device=args.device, agnostic_nms=True, iou=.45, max_det=100,
                       project="runs", name="mask_test", plots=True)
    record_metrics(test, best, "test", evidence / "test-metrics.json")
    exported = Path(trained.export(format="onnx", imgsz=args.imgsz, batch=1,
                                    dynamic=False, simplify=False, opset=17,
                                    nms=False, half=False, device="cpu"))
    # Consolidate external tensor files if any, so the deployment model is one file.
    graph = onnx.load(str(exported), load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(graph)
    onnx.checker.check_model(graph)
    target = Path("assets/model.onnx")
    target.parent.mkdir(exist_ok=True)
    onnx.save_model(graph, str(target), save_as_external_data=False)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    output = session.run(None, {spec.name: np.zeros((1, 3, args.imgsz, args.imgsz), np.float32)})[0]
    if spec.shape != [1, 3, args.imgsz, args.imgsz] or output.shape[:2] != (1, 7):
        raise ValueError(f"Wrong exported contract: input={spec.shape}, output={output.shape}")
    onnx_metrics = YOLO(str(target), task="detect").val(
        data=str(runtime_yaml.resolve()), split="test", imgsz=args.imgsz, batch=1,
        device="cpu", agnostic_nms=True, iou=.45, max_det=100, project="runs", name="mask_onnx_test", plots=True)
    record_metrics(onnx_metrics, target, "test", evidence / "onnx-test-metrics.json")
    # Three non-square test frames check custom resize/decode against the export runtime.
    from PIL import Image
    from detector import Detector
    detector = Detector(target)
    test_dir = Path(config["path"]) / config["test"]
    parity = []
    reference = YOLO(str(target), task="detect")
    for file in sorted(test_dir.glob("*"))[:3]:
        with Image.open(file) as source:
            image = source.convert("RGB")
        custom, _ = detector.predict(image)
        ref = reference.predict(str(file), imgsz=args.imgsz, conf=.4, iou=.45,
                                agnostic_nms=True, max_det=100, device="cpu", verbose=False)[0]
        parity.append({"image": file.name, "custom": custom,
                       "reference": [{"class_id": int(c), "confidence": float(s),
                                      "box": b.tolist()}
                                     for b, c, s in zip(ref.boxes.xyxy.cpu().numpy(),
                                                       ref.boxes.cls.cpu().numpy(),
                                                       ref.boxes.conf.cpu().numpy())]})
    (evidence / "export-parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    (evidence / "model-info.json").write_text(json.dumps({"model": str(target),
        "sha256": checksum, "bytes": target.stat().st_size, "input": spec.shape,
        "output": list(output.shape), "classes": NAMES, "ultralytics": "8.3.230"}, indent=2), encoding="utf-8")
    shutil.copy2(best, evidence / "best.pt")
    print(f"Exported {target}; SHA256={checksum}")


if __name__ == "__main__":
    main()
