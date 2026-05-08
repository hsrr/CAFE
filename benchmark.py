import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import FeatureDataset
from model import DetectionModule, SimilarityModule

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


class EndToEndClassifier(nn.Module):
    """Wrap the two-stage pipeline into a single forward pass."""

    def __init__(self, similarity_module: SimilarityModule, detection_module: DetectionModule):
        super().__init__()
        self.similarity_module = similarity_module
        self.detection_module = detection_module

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        text_aligned, image_aligned, _ = self.similarity_module(text, image)
        return self.detection_module(text, image, text_aligned, image_aligned)


class SyntheticFeatureDataset(Dataset):
    """Fallback dataset for profiling when real data is unavailable."""

    def __init__(self, num_samples: int, seq_len: int, text_dim: int, image_dim: int, num_classes: int):
        self.text = torch.randn(num_samples, seq_len, text_dim, dtype=torch.float32)
        self.image = torch.randn(num_samples, image_dim, dtype=torch.float32)
        self.label = torch.randint(low=0, high=max(num_classes, 2), size=(num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.text.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.text[index], self.image[index], self.label[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark classification inference for the current multimodal model."
    )
    parser.add_argument("--text-npz", type=str, default=None, help="Path to text npz file.")
    parser.add_argument("--image-npz", type=str, default=None, help="Path to image npz file.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, e.g. cpu or cuda:0.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for throughput evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument(
        "--single-sample-steps",
        type=int,
        default=50,
        help="Measured iterations for single-sample forward latency.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap for dataset batches. 0 means use all batches.",
    )
    parser.add_argument(
        "--flops-backend",
        type=str,
        default="auto",
        choices=("auto", "thop", "fvcore", "none"),
        help="Backend used for static compute analysis.",
    )
    parser.add_argument(
        "--combined-checkpoint",
        type=str,
        default=None,
        help="Checkpoint containing the entire wrapped model or nested similarity/detection state dicts.",
    )
    parser.add_argument(
        "--similarity-checkpoint",
        type=str,
        default=None,
        help="Checkpoint for SimilarityModule.",
    )
    parser.add_argument(
        "--detection-checkpoint",
        type=str,
        default=None,
        help="Checkpoint for DetectionModule.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Classifier output dimension.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=256,
        help="Synthetic sample count when no dataset files are provided.",
    )
    parser.add_argument(
        "--synthetic-seq-len",
        type=int,
        default=32,
        help="Synthetic text sequence length.",
    )
    parser.add_argument(
        "--synthetic-text-dim",
        type=int,
        default=200,
        help="Synthetic text feature dimension.",
    )
    parser.add_argument(
        "--synthetic-image-dim",
        type=int,
        default=512,
        help="Synthetic image feature dimension.",
    )
    parser.add_argument(
        "--synthetic-num-classes",
        type=int,
        default=None,
        help="Synthetic label classes for fallback data.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump all metrics as JSON.",
    )
    return parser.parse_args()


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def is_tensor_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, dict) and candidate and all(torch.is_tensor(v) for v in candidate.values())


def extract_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    if is_tensor_state_dict(checkpoint_obj):
        return strip_module_prefix(checkpoint_obj)

    if isinstance(checkpoint_obj, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "network",
            "module",
        ):
            candidate = checkpoint_obj.get(key)
            if is_tensor_state_dict(candidate):
                return strip_module_prefix(candidate)

    raise ValueError("Unable to locate a valid state_dict in checkpoint.")


def load_module_checkpoint(module: nn.Module, checkpoint_path: str, nested_key: Optional[str] = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if nested_key and isinstance(checkpoint, dict) and nested_key in checkpoint:
        state_dict = extract_state_dict(checkpoint[nested_key])
    else:
        state_dict = extract_state_dict(checkpoint)
    module.load_state_dict(state_dict, strict=True)


def maybe_load_checkpoints(
    model: EndToEndClassifier,
    combined_checkpoint: Optional[str],
    similarity_checkpoint: Optional[str],
    detection_checkpoint: Optional[str],
) -> List[str]:
    loaded = []

    if combined_checkpoint:
        checkpoint = torch.load(combined_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "similarity_module" in checkpoint and "detection_module" in checkpoint:
            model.similarity_module.load_state_dict(
                extract_state_dict(checkpoint["similarity_module"]),
                strict=True,
            )
            model.detection_module.load_state_dict(
                extract_state_dict(checkpoint["detection_module"]),
                strict=True,
            )
        else:
            model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        loaded.append(f"combined:{combined_checkpoint}")

    if similarity_checkpoint:
        load_module_checkpoint(model.similarity_module, similarity_checkpoint)
        loaded.append(f"similarity:{similarity_checkpoint}")

    if detection_checkpoint:
        load_module_checkpoint(model.detection_module, detection_checkpoint)
        loaded.append(f"detection:{detection_checkpoint}")

    return loaded


def create_model(device: torch.device, num_classes: int) -> EndToEndClassifier:
    similarity_module = SimilarityModule()
    detection_module = DetectionModule(num_classes=num_classes)
    model = EndToEndClassifier(similarity_module, detection_module)
    model.eval()
    model.to(device)
    return model


def create_dataset(args: argparse.Namespace) -> Tuple[Dataset, str]:
    if args.text_npz and args.image_npz:
        dataset = FeatureDataset(args.text_npz, args.image_npz)
        return dataset, "real"

    synthetic_num_classes = args.synthetic_num_classes or args.num_classes
    dataset = SyntheticFeatureDataset(
        num_samples=args.synthetic_samples,
        seq_len=args.synthetic_seq_len,
        text_dim=args.synthetic_text_dim,
        image_dim=args.synthetic_image_dim,
        num_classes=synthetic_num_classes,
    )
    return dataset, "synthetic"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(values_ms: Sequence[float]) -> Dict[str, float]:
    values = list(values_ms)
    if not values:
        return {
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    return {
        "mean_ms": float(statistics.mean(values)),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
    }


def get_cuda_memory_metrics(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }

    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "cuda_max_memory_allocated_mb": float(allocated),
        "cuda_max_memory_reserved_mb": float(reserved),
    }


def move_batch_to_device(batch: Sequence[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


def profile_compute_cost(
    model: nn.Module,
    sample_inputs: Tuple[torch.Tensor, torch.Tensor],
    backend: str,
) -> Dict[str, Any]:
    selected_backend = backend
    if backend == "auto":
        if thop_profile is not None:
            selected_backend = "thop"
        elif FlopCountAnalysis is not None:
            selected_backend = "fvcore"
        else:
            selected_backend = "none"

    result = {
        "backend": selected_backend,
        "forward_macs": None,
        "forward_flops": None,
        "notes": [],
    }

    if selected_backend == "none":
        result["notes"].append("Install thop or fvcore to enable static compute analysis.")
        return result

    with torch.inference_mode():
        if selected_backend == "thop":
            if thop_profile is None:
                raise RuntimeError("thop is not installed.")
            macs, _ = thop_profile(model, inputs=sample_inputs, verbose=False)
            result["forward_macs"] = float(macs)
            result["forward_flops"] = float(macs * 2.0)
            result["notes"].append("FLOPs are reported as 2 x MACs under the thop convention.")
            return result

        if selected_backend == "fvcore":
            if FlopCountAnalysis is None:
                raise RuntimeError("fvcore is not installed.")
            analysis = FlopCountAnalysis(model, sample_inputs)
            flops = analysis.total()
            unsupported = analysis.unsupported_ops()
            result["forward_macs"] = float(flops)
            result["forward_flops"] = float(flops * 2.0)
            if unsupported:
                result["notes"].append(
                    "Unsupported ops were skipped by fvcore: "
                    + ", ".join(f"{name} x {count}" for name, count in unsupported.items())
                )
            result["notes"].append("FLOPs are approximated as 2 x fvcore-reported ops.")
            return result

    raise ValueError(f"Unsupported backend: {selected_backend}")


def benchmark_single_sample_latency(
    model: nn.Module,
    sample_inputs: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    timings_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(max(warmup_steps, 0)):
            _ = model(*sample_inputs)
        maybe_sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(max(measure_steps, 1)):
            maybe_sync(device)
            start = time.perf_counter()
            _ = model(*sample_inputs)
            maybe_sync(device)
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)

    metrics = summarize_ms(timings_ms)
    metrics.update(get_cuda_memory_metrics(device))
    return metrics


def benchmark_dataloader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, Any]:
    forward_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []
    per_sample_forward_ms: List[float] = []
    batch_sizes: List[int] = []
    total_samples = 0
    total_e2e_seconds = 0.0

    iterator = iter(loader)
    batch_index = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        while True:
            if max_batches and batch_index >= max_batches:
                break

            e2e_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break

            text, image, label = move_batch_to_device(batch, device)
            del label
            maybe_sync(device)
            forward_start = time.perf_counter()
            _ = model(text, image)
            maybe_sync(device)
            forward_end = time.perf_counter()
            e2e_end = time.perf_counter()

            batch_size = int(text.shape[0])
            forward_ms = (forward_end - forward_start) * 1000.0
            e2e_ms = (e2e_end - e2e_start) * 1000.0

            forward_latencies_ms.append(forward_ms)
            e2e_latencies_ms.append(e2e_ms)
            per_sample_forward_ms.append(forward_ms / max(batch_size, 1))
            batch_sizes.append(batch_size)
            total_samples += batch_size
            total_e2e_seconds += max(e2e_end - e2e_start, 0.0)
            batch_index += 1

    return {
        "num_batches_measured": batch_index,
        "num_samples_measured": total_samples,
        "generate_batch_latency_ms": summarize_ms(forward_latencies_ms),
        "e2e_batch_latency_ms": summarize_ms(e2e_latencies_ms),
        "generate_sample_latency_mean_ms": float(statistics.mean(per_sample_forward_ms))
        if per_sample_forward_ms
        else float("nan"),
        "eval_samples_per_sec": float(total_samples / total_e2e_seconds)
        if total_e2e_seconds > 0
        else float("nan"),
        "cuda_memory_peak": get_cuda_memory_metrics(device),
        "batch_size_mean": float(statistics.mean(batch_sizes)) if batch_sizes else float("nan"),
    }


def build_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{value:.4f}"


def print_report(report: Dict[str, Any]) -> None:
    parameter_metrics = report["parameters"]
    compute_metrics = report["compute_cost"]
    single_metrics = report["single_sample_latency_ms"]
    batch_metrics = report["throughput"]
    memory_metrics = report["vram_memory_peak"]

    print("=== Benchmark Setup ===")
    print(f"device: {report['setup']['device']}")
    print(f"dataset_mode: {report['setup']['dataset_mode']}")
    print(f"checkpoint_sources: {report['setup']['checkpoint_sources']}")
    print("")

    print("=== Compute Cost & Parameters ===")
    print(f"params_total: {parameter_metrics['params_total']}")
    print(f"params_trainable: {parameter_metrics['params_trainable']}")
    print(f"flops_backend: {compute_metrics['backend']}")
    print(f"forward_macs: {compute_metrics['forward_macs']}")
    print(f"forward_flops: {compute_metrics['forward_flops']}")
    if compute_metrics["notes"]:
        print("notes:")
        for note in compute_metrics["notes"]:
            print(f"  - {note}")
    print("")

    print("=== Single-Sample Forward Latency ===")
    print(f"mean_ms: {format_number(single_metrics['mean_ms'])}")
    print(f"p95_ms: {format_number(single_metrics['p95_ms'])}")
    print(f"p50_ms: {format_number(single_metrics['p50_ms'])}")
    print(f"std_ms: {format_number(single_metrics['std_ms'])}")
    print("")

    print("=== Throughput & Batch Latency ===")
    print(f"num_batches_measured: {batch_metrics['num_batches_measured']}")
    print(f"num_samples_measured: {batch_metrics['num_samples_measured']}")
    print(
        "generate_batch_latency_mean_ms: "
        f"{format_number(batch_metrics['generate_batch_latency_ms']['mean_ms'])}"
    )
    print(
        "generate_batch_latency_p95_ms: "
        f"{format_number(batch_metrics['generate_batch_latency_ms']['p95_ms'])}"
    )
    print(
        "e2e_batch_latency_mean_ms: "
        f"{format_number(batch_metrics['e2e_batch_latency_ms']['mean_ms'])}"
    )
    print(
        "e2e_batch_latency_p95_ms: "
        f"{format_number(batch_metrics['e2e_batch_latency_ms']['p95_ms'])}"
    )
    print(
        "generate_sample_latency_mean_ms: "
        f"{format_number(batch_metrics['generate_sample_latency_mean_ms'])}"
    )
    print(f"eval_samples_per_sec: {format_number(batch_metrics['eval_samples_per_sec'])}")
    print("")

    print("=== VRAM Memory Peak ===")
    print(
        "cuda_max_memory_allocated_mb: "
        f"{format_number(memory_metrics['cuda_max_memory_allocated_mb'])}"
    )
    print(
        "cuda_max_memory_reserved_mb: "
        f"{format_number(memory_metrics['cuda_max_memory_reserved_mb'])}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset, dataset_mode = create_dataset(args)
    loader = build_loader(dataset, args, device)
    first_text, first_image, _ = dataset[0]
    sample_inputs = (
        first_text.unsqueeze(0).to(device),
        first_image.unsqueeze(0).to(device),
    )

    model = create_model(device, num_classes=args.num_classes)
    checkpoint_sources = maybe_load_checkpoints(
        model,
        combined_checkpoint=args.combined_checkpoint,
        similarity_checkpoint=args.similarity_checkpoint,
        detection_checkpoint=args.detection_checkpoint,
    )

    parameters = count_parameters(model)
    compute_cost = profile_compute_cost(model, sample_inputs, args.flops_backend)
    single_sample_latency = benchmark_single_sample_latency(
        model,
        sample_inputs,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.single_sample_steps,
    )
    throughput = benchmark_dataloader(
        model,
        loader,
        device=device,
        max_batches=args.max_batches,
    )

    vram_memory_peak = throughput["cuda_memory_peak"]
    report = {
        "setup": {
            "device": str(device),
            "dataset_mode": dataset_mode,
            "checkpoint_sources": checkpoint_sources,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "single_sample_steps": args.single_sample_steps,
            "max_batches": args.max_batches,
        },
        "parameters": parameters,
        "compute_cost": compute_cost,
        "single_sample_latency_ms": single_sample_latency,
        "throughput": throughput,
        "vram_memory_peak": vram_memory_peak,
    }

    print_report(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
