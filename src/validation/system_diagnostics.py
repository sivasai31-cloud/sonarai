"""SONARIS-X System Diagnostics Module.

Provides system diagnostics including:
- model size
- inference time
- total pipeline time
- CPU usage
- memory usage

Creates optimization path for ONNX, quantization, edge deployment.
"""

import time
import os
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SystemDiagnostics:
    """System performance diagnostics."""
    model_size_mb: float = 0.0
    inference_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    memory_available_mb: float = 0.0
    optimization_path: Dict[str, str] = field(default_factory=dict)


class SystemDiagnosticsCollector:
    """Collects system performance metrics."""

    def __init__(self):
        self._last_diagnostics: Optional[SystemDiagnostics] = None

    def collect(self, model_path: Optional[str] = None,
                inference_time_ms: float = 0.0,
                pipeline_time_ms: float = 0.0) -> SystemDiagnostics:
        """Collect complete system diagnostics."""
        import torch

        # Model size
        model_size_mb = 0.0
        if model_path and os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            try:
                if torch.cuda.is_available():
                    model_size_mb = 0.0  # Unknown without path
            except Exception:
                pass

        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.used / (1024 * 1024)
        memory_available = memory.available / (1024 * 1024)

        # GPU info
        gpu_available = False
        gpu_name = ""
        try:
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        # Optimization path
        optimization_path = self._get_optimization_path(model_size_mb, gpu_available)

        diagnostics = SystemDiagnostics(
            model_size_mb=round(model_size_mb, 2),
            inference_time_ms=inference_time_ms,
            pipeline_time_ms=round(pipeline_time_ms, 2),
            cpu_usage_percent=round(cpu_usage, 1),
            memory_usage_mb=round(memory_usage, 1),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            memory_available_mb=round(memory_available, 1),
            optimization_path=optimization_path
        )

        self._last_diagnostics = diagnostics
        return diagnostics

    def _get_optimization_path(self, model_size_mb: float,
                                  gpu_available: bool) -> Dict[str, str]:
        """Get optimization path for deployment."""
        path = {}

        if model_size_mb > 100:
            path['recommendation'] = 'Consider model compression'
            path['onnx_export'] = 'torch.onnx.export(model, dummy_input, "model.onnx")'
            path['quantization'] = 'torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)'
        elif model_size_mb > 50:
            path['recommendation'] = 'ONNX export recommended for edge'
            path['onnx_export'] = 'torch.onnx.export(model, dummy_input, "model.onnx")'
        else:
            path['recommendation'] = 'Model is suitable for edge deployment'

        if gpu_available:
            path['deployment'] = 'GPU-accelerated inference available'
        else:
            path['deployment'] = 'CPU inference - consider INT8 quantization for edge'

        path['edge_readiness'] = 'Model can be exported to ONNX format for edge deployment'
        path['quantization'] = 'Dynamic quantization can reduce model size by 4x with minimal accuracy loss'

        return path

    def print_report(self, diagnostics: Optional[SystemDiagnostics] = None):
        """Print system diagnostics report."""
        d = diagnostics or self._last_diagnostics
        if not d:
            print("No diagnostics collected yet.")
            return

        lines = [
            "=" * 60,
            "SONARIS-X SYSTEM DIAGNOSTICS",
            "=" * 60,
            f"Model Size: {d.model_size_mb:.2f} MB",
            f"Inference Time: {d.inference_time_ms:.2f} ms",
            f"Pipeline Time: {d.pipeline_time_ms:.2f} ms",
            f"CPU Usage: {d.cpu_usage_percent:.1f}%",
            f"Memory Usage: {d.memory_usage_mb:.1f} MB",
            f"Memory Available: {d.memory_available_mb:.1f} MB",
            f"GPU Available: {d.gpu_available}",
            f"GPU Name: {d.gpu_name}",
            "",
            "OPTIMIZATION PATH:",
            f"  Recommendation: {d.optimization_path.get('recommendation', 'N/A')}",
            f"  Deployment: {d.optimization_path.get('deployment', 'N/A')}",
            f"  Quantization: {d.optimization_path.get('quantization', 'N/A')}",
            "=" * 60
        ]
        return "\n".join(lines)
