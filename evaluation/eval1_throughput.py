import os
import argparse
import sys
import torch
import time
import csv
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models.model_loader import get_model_with_head
from load_models.TempScaleWrapper import TempScaleWrapper
from load_models.ShallowEnsembleWrapper import ShallowEnsembleWrapper
from load_models.DuoWrapper import DuoWrapper

def benchmark_model(model, input_tensor, device, warmup=10, reps=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_latency = total_time / reps
    throughput = input_tensor.size(0) / avg_latency
    return avg_latency, throughput

def main(model_name, source, num_classes, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    results = []

    # Base model
    model, _ = get_model_with_head(
        model_name=model_name,
        num_classes=num_classes,
        source=source,
        freeze=True,
        m_head=1
    )
    model = model.to(device)
    latency, throughput = benchmark_model(model, dummy_input, device)
    print(f"{model_name} ({source}) - Base: {latency:.4f}s | {throughput:.2f} samples/sec")
    results.append([model_name, source, "base", 1, batch_size, latency, throughput])

    # TempScaleWrapper
    ts_model = TempScaleWrapper(model).to(device)
    latency_ts, throughput_ts = benchmark_model(ts_model, dummy_input, device)
    print(f"{model_name} ({source}) - TempScale: {latency_ts:.4f}s | {throughput_ts:.2f} samples/sec")
    results.append([model_name, source, "temp_scale", 1, batch_size, latency_ts, throughput_ts])

    # Shallow ensemble
    model_2h, _ = get_model_with_head(
        model_name=model_name,
        num_classes=num_classes,
        source=source,
        freeze=True,
        m_head=2
    )
    se_model = ShallowEnsembleWrapper(model_2h.to(device))
    latency_se, throughput_se = benchmark_model(se_model, dummy_input, device)
    print(f"{model_name} ({source}) - Shallow Ensemble: {latency_se:.4f}s | {throughput_se:.2f} samples/sec")
    results.append([model_name, source, "shallow_ensemble", 2, batch_size, latency_se, throughput_se])

    # DuoWrapper
    duo_model = DuoWrapper(model_large=TempScaleWrapper(model), model_small=TempScaleWrapper(model)).to(device)
    latency_duo, throughput_duo = benchmark_model(duo_model, dummy_input, device)
    print(f"{model_name} ({source}) - DuoWrapper: {latency_duo:.4f}s | {throughput_duo:.2f} samples/sec")
    results.append([model_name, source, "duo", 2, batch_size, latency_duo, throughput_duo])

    # Save to CSV
    os.makedirs("result", exist_ok=True)
    csv_path = "result/throughput.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_name", "source", "wrapper", "m_head", "batch_size", "latency_sec", "throughput_samples_per_sec"])
        writer.writerows(results)
    print(f"✅ Saved throughput results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--source")
    parser.add_argument("--num_classes", type=int, default=257)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.model_name, args.source, args.num_classes, args.batch_size)