# ============================================================
# Model presets — popular open-weight models with sensible defaults
# ============================================================

locals {
  model_presets = {
    # --- Gemma family ---
    "gemma-2-2b-it" = {
      model_id              = "google/gemma-2-2b-it"
      machine_type          = "g2-standard-4"
      max_model_len         = 4096
      gpu_memory_utilization = 0.85
      gated                 = true
      multimodal            = false
    }
    "gemma-2-9b-it" = {
      model_id              = "google/gemma-2-9b-it"
      machine_type          = "g2-standard-4"
      max_model_len         = 4096
      gpu_memory_utilization = 0.90
      gated                 = true
      multimodal            = false
    }
    "gemma-3-4b-it" = {
      model_id              = "google/gemma-3-4b-it"
      machine_type          = "g2-standard-4"
      max_model_len         = 8192
      gpu_memory_utilization = 0.85
      gated                 = true
      multimodal            = true
    }
    "gemma-3-12b-it" = {
      model_id              = "google/gemma-3-12b-it"
      machine_type          = "g2-standard-8"
      max_model_len         = 8192
      gpu_memory_utilization = 0.90
      gated                 = true
      multimodal            = true
    }
    # --- Llama family ---
    "llama-3.2-1b-instruct" = {
      model_id              = "meta-llama/Llama-3.2-1B-Instruct"
      machine_type          = "g2-standard-4"
      max_model_len         = 4096
      gated                 = true
    }
    "llama-3.2-3b-instruct" = {
      model_id              = "meta-llama/Llama-3.2-3B-Instruct"
      machine_type          = "g2-standard-4"
      max_model_len         = 4096
      gated                 = true
    }
    # --- Qwen family ---
    "qwen2.5-3b-instruct" = {
      model_id              = "Qwen/Qwen2.5-3B-Instruct"
      machine_type          = "g2-standard-4"
      max_model_len         = 8192
      gated                 = false
    }
    "qwen2.5-7b-instruct" = {
      model_id              = "Qwen/Qwen2.5-7B-Instruct"
      machine_type          = "g2-standard-4"
      max_model_len         = 8192
      gpu_memory_utilization = 0.90
      gated                 = false
    }
    "qwen2.5-14b-instruct" = {
      model_id              = "Qwen/Qwen2.5-14B-Instruct"
      machine_type          = "g2-standard-8"
      max_model_len         = 8192
      gpu_memory_utilization = 0.90
      gated                 = false
    }
    # --- Mistral family ---
    "mistral-7b-instruct" = {
      model_id              = "mistralai/Mistral-7B-Instruct-v0.3"
      machine_type          = "g2-standard-4"
      max_model_len         = 32768
      gpu_memory_utilization = 0.90
      gated                 = false
    }
    # --- Microsoft Phi ---
    "phi-3.5-mini-instruct" = {
      model_id              = "microsoft/Phi-3.5-mini-instruct"
      machine_type          = "g2-standard-4"
      max_model_len         = 4096
      gated                 = false
    }
    # --- TinyLlama (ultra-small, fast boot) ---
    "tinyllama-1.1b-chat" = {
      model_id              = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      machine_type          = "g2-standard-4"
      max_model_len         = 2048
      gated                 = false
    }
    # ================================================================
    # LARGE MODELS — multi-GPU, A100/H100, expensive
    # ================================================================
    "mixtral-8x7b-instruct" = {
      model_id              = "mistralai/Mixtral-8x7B-Instruct-v0.1"
      machine_type          = "a2-ultragpu-2g"  # 2x A100 80GB = 160GB
      max_model_len         = 32768
      gpu_memory_utilization = 0.90
      tensor_parallel_size  = 2
      disk_size_gb          = 300
      gated                 = false
    }
    "qwen2.5-72b-instruct" = {
      model_id              = "Qwen/Qwen2.5-72B-Instruct"
      machine_type          = "a2-ultragpu-2g"  # 2x A100 80GB = 160GB
      max_model_len         = 32768
      gpu_memory_utilization = 0.92
      tensor_parallel_size  = 2
      disk_size_gb          = 300
      gated                 = false
    }
    "mixtral-8x22b-instruct" = {
      model_id              = "mistralai/Mixtral-8x22B-Instruct-v0.1"
      machine_type          = "a2-ultragpu-4g"  # 4x A100 80GB = 320GB
      max_model_len         = 32768
      gpu_memory_utilization = 0.92
      tensor_parallel_size  = 4
      disk_size_gb          = 500
      gated                 = false
    }
    "deepseek-v3" = {
      model_id              = "deepseek-ai/DeepSeek-V3"
      machine_type          = "a3-highgpu-8g"   # 8x H100 80GB = 640GB
      max_model_len         = 32768
      gpu_memory_utilization = 0.95
      tensor_parallel_size  = 8
      disk_size_gb          = 1000
      extra_vllm_args       = "--trust-remote-code"
      gated                 = false
    }
    "deepseek-r1" = {
      model_id              = "deepseek-ai/DeepSeek-R1"
      machine_type          = "a3-highgpu-8g"   # 8x H100 80GB = 640GB
      max_model_len         = 32768
      gpu_memory_utilization = 0.95
      tensor_parallel_size  = 8
      disk_size_gb          = 1000
      extra_vllm_args       = "--trust-remote-code"
      gated                 = false
    }
    "kimi-k3" = {
      model_id              = "moonshotai/Kimi-K3"
      machine_type          = "a3-highgpu-8g"   # 8x H100 80GB = 640GB (see note)
      max_model_len         = 131072
      gpu_memory_utilization = 0.95
      tensor_parallel_size  = 8
      disk_size_gb          = 2000
      extra_vllm_args       = "--trust-remote-code --quantization mxfp4"
      gated                 = false
    }
    "llama-3.1-405b-instruct" = {
      model_id              = "meta-llama/Llama-3.1-405B-Instruct"
      machine_type          = "a3-highgpu-8g"   # 8x H100 80GB = 640GB (needs FP8)
      max_model_len         = 32768
      gpu_memory_utilization = 0.95
      tensor_parallel_size  = 8
      disk_size_gb          = 1500
      extra_vllm_args       = "--quantization fp8"
      gated                 = true
    }
  }
}

# ============================================================
# Variables
# ============================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "aldianfazrihady"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-b"
}

variable "model_preset" {
  description = <<-EOT
    Preset name from the catalog. Use `terraform output` to see all options.
    Popular: qwen2.5-3b-instruct, mistral-7b-instruct, gemma-3-4b-it, etc.
    Leave empty and set model_id to serve any arbitrary HuggingFace model.
    EOT
  type        = string
  default     = "qwen2.5-3b-instruct"
}

variable "model_id" {
  description = "Direct HuggingFace model ID (overrides preset). Example: Qwen/Qwen2.5-3B-Instruct"
  type        = string
  default     = null
}

variable "machine_type" {
  description = "GCE machine type. Overrides preset. g2-standard-4=1xL4(24GB), g2-standard-8=1xL4+more CPU/RAM, g2-standard-24=2xL4(48GB)."
  type        = string
  default     = null
}

variable "max_model_len" {
  description = "Maximum sequence length. Overrides preset."
  type        = number
  default     = null
}

variable "gpu_memory_utilization" {
  description = "Fraction of GPU memory vLLM uses (0.0-1.0). Overrides preset."
  type        = number
  default     = null
}

variable "vllm_port" {
  description = "Port for vLLM OpenAI-compatible API"
  type        = number
  default     = 0
}

variable "vllm_image" {
  description = "vLLM Docker image to use"
  type        = string
  default     = "vllm/vllm-openai:latest"
}

variable "tensor_parallel_size" {
  description = "Number of GPUs to split the model across (tensor parallelism). Auto-set by presets; override for custom models."
  type        = number
  default     = null
}

variable "disk_size_gb" {
  description = "Boot disk size in GB. Large models need bigger disks for weight downloads."
  type        = number
  default     = null
}

variable "extra_vllm_args" {
  description = "Additional vLLM CLI arguments (e.g. '--trust-remote-code --quantization fp8'). Auto-set by some presets."
  type        = string
  default     = null
}

variable "hf_token" {
  description = "HuggingFace token. Required for gated models (Gemma, Llama). Get one at https://huggingface.co/settings/tokens"
  type        = string
  sensitive   = true
  default     = ""
}

# ============================================================
# Computed values — merge preset with user overrides
# ============================================================

locals {
  # Start from preset or empty
  preset = lookup(local.model_presets, var.model_preset, {})

  # If model_id is set directly, it takes priority
  # Otherwise fall back to preset, or error if neither
  effective_model_id = coalesce(
    var.model_id,
    lookup(local.preset, "model_id", ""),
  )

  effective_machine_type = coalesce(var.machine_type, lookup(local.preset, "machine_type", "g2-standard-4"))
  effective_max_model_len = coalesce(var.max_model_len, lookup(local.preset, "max_model_len", 4096))
  effective_gpu_memory_utilization = coalesce(var.gpu_memory_utilization, lookup(local.preset, "gpu_memory_utilization", 0.85))
  effective_tensor_parallel_size = coalesce(var.tensor_parallel_size, lookup(local.preset, "tensor_parallel_size", 1))
  effective_disk_size_gb = coalesce(var.disk_size_gb, lookup(local.preset, "disk_size_gb", 100))
  effective_extra_vllm_args = coalesce(var.extra_vllm_args, lookup(local.preset, "extra_vllm_args", ""))
  effective_port = var.vllm_port == 0 ? 8000 : var.vllm_port

  is_gated     = lookup(local.preset, "gated", false)
  is_multimodal = lookup(local.preset, "multimodal", false)
}
