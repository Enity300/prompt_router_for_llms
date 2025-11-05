import requests
import json
from typing import Dict, Any, Optional
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class LLMClientError(Exception):
    """Custom exception for LLM client errors"""
    pass


def _query_ollama_model(model_name: str, prompt: str, specialist_name: str, category: str, system_prompt: str = None) -> Dict[str, Any]:
   
    if not config.USE_LOCAL_MODELS:
        return None
        
    # GPU monitoring integration
    gpu_metrics_before = None
    gpu_metrics_after = None
    
    try:
        # Try to get GPU metrics before inference
        try:
            from .gpu_monitor import get_gpu_monitor
            gpu_monitor = get_gpu_monitor()
            gpu_metrics_before = gpu_monitor.get_current_metrics()
        except ImportError:
            pass  # GPU monitoring not available
        except Exception:
            pass  # Ignore GPU monitoring errors
        
        start_time = time.time()
        
        # Prepare the request payload for Ollama API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1 if category == "coding" else 0.7,
                "num_predict": 1000
            }
        }
        
        # Make request to local Ollama server
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        response_time = time.time() - start_time
        
        # Get GPU metrics after inference
        try:
            if gpu_metrics_before is not None:
                gpu_metrics_after = gpu_monitor.get_current_metrics()
        except Exception:
            pass  # Ignore GPU monitoring errors
        
        # Calculate GPU usage statistics if available
        gpu_stats = None
        if gpu_metrics_before and gpu_metrics_after:
            try:
                # Calculate GPU utilization changes during inference
                gpu_stats = {}
                for before, after in zip(gpu_metrics_before, gpu_metrics_after):
                    if before.gpu_id == after.gpu_id:
                        gpu_stats[f"gpu_{before.gpu_id}"] = {
                            "name": before.name,
                            "utilization_before": before.utilization_percent,
                            "utilization_after": after.utilization_percent,
                            "utilization_delta": after.utilization_percent - before.utilization_percent,
                            "memory_before": before.memory_percent,
                            "memory_after": after.memory_percent,
                            "memory_delta": after.memory_percent - before.memory_percent,
                            "temperature_before": before.temperature_c,
                            "temperature_after": after.temperature_c,
                            "temperature_delta": after.temperature_c - before.temperature_c
                        }
                        if before.power_watts and after.power_watts:
                            gpu_stats[f"gpu_{before.gpu_id}"]["power_before"] = before.power_watts
                            gpu_stats[f"gpu_{before.gpu_id}"]["power_after"] = after.power_watts
                            gpu_stats[f"gpu_{before.gpu_id}"]["power_delta"] = after.power_watts - before.power_watts
            except Exception:
                pass  # Ignore GPU stats calculation errors
        
        return {
            "specialist": f"{specialist_name} (Local)",
            "category": category,
            "response": data["message"]["content"],
            "tokens_used": data.get("eval_count", 0),
            "model": model_name,
            "success": True,
            "response_time": response_time,
            "is_local": True,
            "gpu_metrics": {
                "gpu_monitoring_available": gpu_metrics_before is not None,
                "gpu_stats": gpu_stats,
                "inference_duration": response_time
            }
        }
        
    except requests.exceptions.ConnectionError:
        print(f"⚠️ Ollama server not running - falling back to API/mock for {specialist_name}")
        return None
    except Exception as e:
        print(f"⚠️ Ollama error for {specialist_name}: {e}")
        return None


def _check_ollama_available() -> bool:
    if not config.USE_LOCAL_MODELS:
        return False
        
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def query_deepseek_coder(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
   
    # Try local Ollama model first
    if config.USE_LOCAL_MODELS:
        local_result = _query_ollama_model(
            model_name=config.LOCAL_CODING_MODEL,
            prompt=prompt,
            specialist_name="DeepSeek Coder 1.3B",
            category="coding",
            system_prompt="You are DeepSeek Coder, an expert coding assistant specialized in programming tasks, code generation, debugging, and software development. Provide clear, working code solutions."
        )
        if local_result:
            return local_result
    
    # Fallback to API or mock
    if not config.DEEPSEEK_API_KEY:
        return {
            "specialist": "DeepSeek Coder (API Key Required)",
            "category": "coding",
            "response": f"DeepSeek API key not configured. Add DEEPSEEK_API_KEY to your .env file to use this specialist.",
            "tokens_used": 0,
            "model": "deepseek-coder",
            "success": False,
            "error": "API key not configured"
        }
    
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-coder",
        "messages": [
            {"role": "system", "content": "You are DeepSeek Coder, an expert coding assistant specialized in programming tasks, code generation, debugging, and software development."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{config.DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "specialist": "DeepSeek Coder",
            "category": "coding",
            "response": data["choices"][0]["message"]["content"],
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
            "model": "deepseek-coder",
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        raise LLMClientError(f"DeepSeek API error: {e}")
    except (KeyError, IndexError) as e:
        raise LLMClientError(f"DeepSeek response parsing error: {e}")


def query_wizardmath(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    # Try local Ollama model first
    if config.USE_LOCAL_MODELS:
        local_result = _query_ollama_model(
            model_name=config.LOCAL_MATH_MODEL,
            prompt=prompt,
            specialist_name="Qwen2-Math 1.5B",
            category="math",
            system_prompt="You are Qwen2-Math, an expert mathematical assistant specialized in solving mathematical problems, equations, and providing step-by-step mathematical reasoning."
        )
        if local_result:
            return local_result
    
    # Fallback to API or mock
    if not config.WIZARDMATH_API_KEY:
        return {
        "specialist": "WizardMath (API Key Required)",
        "category": "math",
        "response": f"WizardMath API key not configured. Add WIZARDMATH_API_KEY to your .env file to use this specialist.",
        "tokens_used": 0,
        "model": "wizardmath-70b",
        "success": False,
        "error": "API key not configured"
    }
    
    headers = {
        "Authorization": f"Bearer {config.WIZARDMATH_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "wizardmath-70b",
        "messages": [
            {"role": "system", "content": "You are WizardMath, an expert mathematical assistant specialized in solving mathematical problems, equations, calculus, statistics, and mathematical reasoning."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{config.WIZARDMATH_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "specialist": "WizardMath",
            "category": "math", 
            "response": data["choices"][0]["message"]["content"],
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
            "model": "wizardmath-70b",
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        raise LLMClientError(f"WizardMath API error: {e}")
    except (KeyError, IndexError) as e:
        raise LLMClientError(f"WizardMath response parsing error: {e}")


def query_openai_gpt4(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    # Try local Ollama model first
    if config.USE_LOCAL_MODELS:
        local_result = _query_ollama_model(
            model_name=config.LOCAL_GENERAL_MODEL,
            prompt=prompt,
            specialist_name="Llama 3.2 1B",
            category="general_knowledge",
            system_prompt="You are Llama 3.2, a helpful and knowledgeable AI assistant capable of handling a wide variety of tasks and questions across different domains."
        )
        if local_result:
            return local_result
    
    # Fallback to API or mock
    if not config.OPENAI_API_KEY:
        return {
        "specialist": "OpenAI GPT-4 (API Key Required)",
        "category": "general_knowledge",
        "response": f"OpenAI API key not configured. Add OPENAI_API_KEY to your .env file to use this specialist.",
        "tokens_used": 0,
        "model": "gpt-4",
        "success": False,
        "error": "API key not configured"
    }
    
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are GPT-4, a helpful and knowledgeable AI assistant capable of handling a wide variety of tasks and questions across different domains."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{config.OPENAI_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "specialist": "OpenAI GPT-4",
            "category": "general_knowledge",
            "response": data["choices"][0]["message"]["content"],
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
            "model": "gpt-4",
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        raise LLMClientError(f"OpenAI API error: {e}")
    except (KeyError, IndexError) as e:
        raise LLMClientError(f"OpenAI response parsing error: {e}")


def query_default_model(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    # Try local fallback model first (Phi3:mini)
    if config.USE_LOCAL_MODELS:
        local_result = _query_ollama_model(
            model_name=config.LOCAL_FALLBACK_MODEL,
            prompt=prompt,
            specialist_name="Phi3 Mini",
            category="general_knowledge",
            system_prompt="You are Phi3-Mini, a helpful and efficient AI assistant designed to handle diverse queries with clear, concise responses."
        )
        if local_result:
            return local_result
    
    # Fallback to OpenAI GPT-4 (which has its own local model logic)
    return query_openai_gpt4(prompt, max_tokens)


# Mapping of categories to their respective specialist functions
SPECIALIST_FUNCTIONS = {
    "coding": query_deepseek_coder,
    "math": query_wizardmath,
    "general_knowledge": query_default_model
}


def get_specialist_function(category: str):
    return SPECIALIST_FUNCTIONS.get(category, query_default_model)


def query_specialist(category: str, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:

    specialist_func = get_specialist_function(category)
    
    start_time = time.time()
    try:
        result = specialist_func(prompt, max_tokens)
        result["response_time"] = time.time() - start_time
        return result
        
    except LLMClientError as e:
        return {
            "specialist": f"Error ({category})",
            "category": category,
            "response": f"Error querying specialist: {e}",
            "tokens_used": 0,
            "model": "error",
            "success": False,
            "error": str(e),
            "response_time": time.time() - start_time
        }
