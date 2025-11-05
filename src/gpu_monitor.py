#!/usr/bin/env python3
"""
SS-GER GPU Monitoring System
============================

Comprehensive GPU monitoring for tracking performance during model inference
and routing operations with academic-grade analysis capabilities.

Features:
- Multi-GPU support with detailed metrics
- Real-time monitoring with background threads
- Routing operation profiling
- Statistical analysis and reporting
- Academic publication-ready data export
"""

import sys
import os
import time
import json
import threading
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU monitoring libraries (optional imports)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@dataclass
class GPUMetrics:
    """GPU performance metrics at a point in time"""
    timestamp: float
    gpu_id: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: float
    power_watts: Optional[float] = None
    fan_percent: Optional[float] = None
    clock_graphics_mhz: Optional[int] = None
    clock_memory_mhz: Optional[int] = None


class GPUMonitor:
    """Comprehensive GPU monitoring for SS-GER system"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history: List[GPUMetrics] = []
        self.results_dir = Path("results/gpu_monitoring")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect available GPU monitoring capabilities
        self.detection_results = self._detect_gpu_capabilities()
        
    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect available GPU monitoring capabilities"""
        capabilities = {
            "gputil_available": GPUTIL_AVAILABLE,
            "nvml_available": NVML_AVAILABLE,
            "pynvml_available": PYNVML_AVAILABLE,
            "nvidia_smi_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "system_info": {}
        }
        
        # Check nvidia-smi availability
        try:
            result = subprocess.run(["nvidia-smi", "-L"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                capabilities["nvidia_smi_available"] = True
                # Parse GPU names from nvidia-smi -L output
                for line in result.stdout.strip().split('\n'):
                    if 'GPU' in line:
                        capabilities["gpu_names"].append(line.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Get GPU count using available methods
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                capabilities["gpu_count"] = len(gpus)
                if not capabilities["gpu_names"]:
                    capabilities["gpu_names"] = [gpu.name for gpu in gpus]
            except Exception:
                pass
        
        if NVML_AVAILABLE and capabilities["gpu_count"] == 0:
            try:
                capabilities["gpu_count"] = nvml.nvmlDeviceGetCount()
                if not capabilities["gpu_names"]:
                    for i in range(capabilities["gpu_count"]):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                        capabilities["gpu_names"].append(name)
            except Exception:
                pass
        
        # Add system info
        if PSUTIL_AVAILABLE:
            capabilities["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": sys.platform
            }
        
        return capabilities
    
    def get_current_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all available GPUs"""
        metrics = []
        
        try:
            if GPUTIL_AVAILABLE:
                metrics.extend(self._get_gputil_metrics())
            elif NVML_AVAILABLE:
                metrics.extend(self._get_nvml_metrics())
            elif self.detection_results["nvidia_smi_available"]:
                metrics.extend(self._get_nvidia_smi_metrics())
        except Exception as e:
            if console:
                console.print(f"[yellow]Warning: Failed to get GPU metrics: {e}[/yellow]")
        
        return metrics
    
    def _get_gputil_metrics(self) -> List[GPUMetrics]:
        """Get metrics using GPUtil library"""
        metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                metric = GPUMetrics(
                    timestamp=time.time(),
                    gpu_id=gpu.id,
                    name=gpu.name,
                    utilization_percent=gpu.load * 100,
                    memory_used_mb=gpu.memoryUsed,
                    memory_total_mb=gpu.memoryTotal,
                    memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    temperature_c=gpu.temperature
                )
                metrics.append(metric)
        except Exception as e:
            if console:
                console.print(f"[yellow]GPUtil error: {e}[/yellow]")
        
        return metrics
    
    def _get_nvml_metrics(self) -> List[GPUMetrics]:
        """Get metrics using NVIDIA ML library"""
        metrics = []
        try:
            device_count = nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Power (optional)
                power = None
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except Exception:
                    pass
                
                # Clock speeds (optional)
                graphics_clock = None
                memory_clock = None
                try:
                    graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                except Exception:
                    pass
                
                metric = GPUMetrics(
                    timestamp=time.time(),
                    gpu_id=i,
                    name=name,
                    utilization_percent=util.gpu,
                    memory_used_mb=mem_info.used / (1024 * 1024),
                    memory_total_mb=mem_info.total / (1024 * 1024),
                    memory_percent=(mem_info.used / mem_info.total) * 100,
                    temperature_c=temp,
                    power_watts=power,
                    clock_graphics_mhz=graphics_clock,
                    clock_memory_mhz=memory_clock
                )
                metrics.append(metric)
                
        except Exception as e:
            if console:
                console.print(f"[yellow]NVML error: {e}[/yellow]")
        
        return metrics
    
    def _get_nvidia_smi_metrics(self) -> List[GPUMetrics]:
        """Get metrics using nvidia-smi command"""
        metrics = []
        try:
            # Query nvidia-smi for detailed metrics
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            try:
                                gpu_id = int(parts[0])
                                name = parts[1]
                                utilization = float(parts[2]) if parts[2] != '[Not Supported]' else 0.0
                                memory_used = float(parts[3]) if parts[3] != '[Not Supported]' else 0.0
                                memory_total = float(parts[4]) if parts[4] != '[Not Supported]' else 1.0
                                temperature = float(parts[5]) if parts[5] != '[Not Supported]' else 0.0
                                power = float(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else None
                                
                                metric = GPUMetrics(
                                    timestamp=time.time(),
                                    gpu_id=gpu_id,
                                    name=name,
                                    utilization_percent=utilization,
                                    memory_used_mb=memory_used,
                                    memory_total_mb=memory_total,
                                    memory_percent=(memory_used / memory_total) * 100 if memory_total > 0 else 0,
                                    temperature_c=temperature,
                                    power_watts=power
                                )
                                metrics.append(metric)
                            except ValueError as e:
                                if console:
                                    console.print(f"[yellow]Error parsing nvidia-smi output: {e}[/yellow]")
                                continue
        except Exception as e:
            if console:
                console.print(f"[yellow]nvidia-smi error: {e}[/yellow]")
        
        return metrics
    
    def start_monitoring(self):
        """Start continuous GPU monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        if console:
            console.print(f"🔍 GPU monitoring started (interval: {self.sample_interval}s)")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        if console:
            console.print("🛑 GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.extend(metrics)
            except Exception as e:
                if console:
                    console.print(f"[yellow]Monitoring error: {e}[/yellow]")
            
            time.sleep(self.sample_interval)
    
    def get_summary_stats(self, last_n_samples: Optional[int] = None) -> Dict[str, Any]:
        """Get summary statistics for monitored period"""
        if not self.metrics_history:
            return {"error": "No monitoring data available"}
        
        recent_metrics = self.metrics_history
        if last_n_samples:
            recent_metrics = self.metrics_history[-last_n_samples:]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Group by GPU
        gpu_stats = {}
        for metric in recent_metrics:
            gpu_id = metric.gpu_id
            if gpu_id not in gpu_stats:
                gpu_stats[gpu_id] = {
                    "name": metric.name,
                    "utilization": [],
                    "memory_percent": [],
                    "temperature": [],
                    "power": []
                }
            
            gpu_stats[gpu_id]["utilization"].append(metric.utilization_percent)
            gpu_stats[gpu_id]["memory_percent"].append(metric.memory_percent)
            gpu_stats[gpu_id]["temperature"].append(metric.temperature_c)
            if metric.power_watts is not None:
                gpu_stats[gpu_id]["power"].append(metric.power_watts)
        
        # Calculate statistics
        summary = {}
        for gpu_id, stats in gpu_stats.items():
            summary[f"gpu_{gpu_id}"] = {
                "name": stats["name"],
                "utilization": {
                    "avg": sum(stats["utilization"]) / len(stats["utilization"]),
                    "max": max(stats["utilization"]),
                    "min": min(stats["utilization"])
                },
                "memory": {
                    "avg": sum(stats["memory_percent"]) / len(stats["memory_percent"]),
                    "max": max(stats["memory_percent"]),
                    "min": min(stats["memory_percent"])
                },
                "temperature": {
                    "avg": sum(stats["temperature"]) / len(stats["temperature"]),
                    "max": max(stats["temperature"]),
                    "min": min(stats["temperature"])
                }
            }
            
            if stats["power"]:
                summary[f"gpu_{gpu_id}"]["power"] = {
                    "avg": sum(stats["power"]) / len(stats["power"]),
                    "max": max(stats["power"]),
                    "min": min(stats["power"])
                }
        
        summary["monitoring_period"] = {
            "duration_seconds": recent_metrics[-1].timestamp - recent_metrics[0].timestamp,
            "samples_count": len(recent_metrics),
            "start_time": datetime.fromtimestamp(recent_metrics[0].timestamp).isoformat(),
            "end_time": datetime.fromtimestamp(recent_metrics[-1].timestamp).isoformat()
        }
        
        return summary
    
    def save_monitoring_data(self, filename: Optional[str] = None) -> str:
        """Save monitoring data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_monitoring_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert metrics to dictionaries for JSON serialization
        data = {
            "detection_results": self.detection_results,
            "monitoring_config": {
                "sample_interval": self.sample_interval,
                "total_samples": len(self.metrics_history)
            },
            "metrics": [asdict(metric) for metric in self.metrics_history],
            "summary_stats": self.get_summary_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def clear_history(self):
        """Clear monitoring history"""
        self.metrics_history.clear()
        if console:
            console.print("📊 GPU monitoring history cleared")
    
    def print_current_status(self):
        """Print current GPU status"""
        metrics = self.get_current_metrics()
        
        if not metrics:
            if console:
                console.print("❌ No GPU metrics available")
                console.print("📋 Detection Results:")
                for key, value in self.detection_results.items():
                    console.print(f"   {key}: {value}")
            else:
                print("❌ No GPU metrics available")
                print("📋 Detection Results:")
                for key, value in self.detection_results.items():
                    print(f"   {key}: {value}")
            return
        
        if console:
            console.print("🎮 Current GPU Status:")
            console.print("=" * 60)
            
            for metric in metrics:
                console.print(f"GPU {metric.gpu_id}: {metric.name}")
                console.print(f"  💪 Utilization: {metric.utilization_percent:.1f}%")
                console.print(f"  🧠 Memory: {metric.memory_used_mb:.0f}MB / {metric.memory_total_mb:.0f}MB ({metric.memory_percent:.1f}%)")
                console.print(f"  🌡️ Temperature: {metric.temperature_c:.1f}°C")
                if metric.power_watts:
                    console.print(f"  ⚡ Power: {metric.power_watts:.1f}W")
                if metric.clock_graphics_mhz:
                    console.print(f"  🕐 Graphics Clock: {metric.clock_graphics_mhz}MHz")
                if metric.clock_memory_mhz:
                    console.print(f"  🕐 Memory Clock: {metric.clock_memory_mhz}MHz")
                console.print()
        else:
            print("🎮 Current GPU Status:")
            print("=" * 60)
            
            for metric in metrics:
                print(f"GPU {metric.gpu_id}: {metric.name}")
                print(f"  💪 Utilization: {metric.utilization_percent:.1f}%")
                print(f"  🧠 Memory: {metric.memory_used_mb:.0f}MB / {metric.memory_total_mb:.0f}MB ({metric.memory_percent:.1f}%)")
                print(f"  🌡️ Temperature: {metric.temperature_c:.1f}°C")
                if metric.power_watts:
                    print(f"  ⚡ Power: {metric.power_watts:.1f}W")
                if metric.clock_graphics_mhz:
                    print(f"  🕐 Graphics Clock: {metric.clock_graphics_mhz}MHz")
                if metric.clock_memory_mhz:
                    print(f"  🕐 Memory Clock: {metric.clock_memory_mhz}MHz")
                print()


class RoutingGPUProfiler:
    """GPU profiler specifically for routing operations"""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.routing_sessions = []
    
    def start_routing_session(self, session_name: str):
        """Start profiling a routing session"""
        self.gpu_monitor.clear_history()
        self.gpu_monitor.start_monitoring()
        
        session = {
            "name": session_name,
            "start_time": time.time(),
            "start_metrics": self.gpu_monitor.get_current_metrics()
        }
        
        self.routing_sessions.append(session)
        if console:
            console.print(f"🎯 Started GPU profiling for: {session_name}")
    
    def end_routing_session(self):
        """End the current routing session and save results"""
        if not self.routing_sessions:
            if console:
                console.print("❌ No active routing session")
            return
        
        self.gpu_monitor.stop_monitoring()
        
        session = self.routing_sessions[-1]
        session["end_time"] = time.time()
        session["end_metrics"] = self.gpu_monitor.get_current_metrics()
        session["duration"] = session["end_time"] - session["start_time"]
        session["summary_stats"] = self.gpu_monitor.get_summary_stats()
        
        # Save session data
        filename = f"routing_profile_{session['name']}_{int(session['start_time'])}.json"
        filepath = self.gpu_monitor.save_monitoring_data(filename)
        
        if console:
            console.print(f"📊 Routing session '{session['name']}' completed")
            console.print(f"   Duration: {session['duration']:.2f} seconds")
            console.print(f"   Profile saved: {filepath}")
        
        return session


# Global GPU monitor instance
_gpu_monitor = None

def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor

