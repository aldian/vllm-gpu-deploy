# vLLM on GCE GPU — Terraform
#
# Provisions a single GPU VM serving ANY open-weight model via vLLM.
# Supports all HuggingFace models: Qwen, Llama, Mistral, Gemma, Phi, and more.
#
# Usage:
#   terraform apply                              # default: qwen2.5-3b-instruct
#   terraform apply -var="model_preset=mistral-7b-instruct"
#   terraform apply -var="model_id=Qwen/Qwen2.5-14B-Instruct"
#
# Destroy everything with: terraform destroy

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Static IP so we have a predictable endpoint
resource "google_compute_address" "static_ip" {
  name         = "vllm-server-ip"
  address_type = "EXTERNAL"
}

# Firewall rule: vLLM API access (lock down source_ranges in production)
resource "google_compute_firewall" "vllm" {
  name    = "vllm-api-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = [local.effective_port]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["vllm-server"]
}

# Firewall rule: SSH
resource "google_compute_firewall" "ssh" {
  name    = "vllm-ssh-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["vllm-server"]
}

# The GPU VM
resource "google_compute_instance" "vllm" {
  name         = "vllm-server"
  machine_type = local.effective_machine_type
  tags         = ["vllm-server"]

  boot_disk {
    initialize_params {
      image = "projects/debian-cloud/global/images/family/debian-12"
      size  = 100
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  service_account {
    email  = google_service_account.vllm.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "true"
  }

  metadata_startup_script = templatefile(
    "${path.module}/startup.sh.tpl",
    {
      model_id               = local.effective_model_id
      vllm_port              = local.effective_port
      vllm_image             = var.vllm_image
      max_model_len          = local.effective_max_model_len
      gpu_memory_utilization = local.effective_gpu_memory_utilization
      hf_token               = var.hf_token
    }
  )
}

resource "google_service_account" "vllm" {
  account_id   = "vllm-server-sa"
  display_name = "vLLM Server Service Account"
}

# ============================================================
# OUTPUTS
# ============================================================

output "vm_external_ip" {
  description = "External IP of the vLLM server"
  value       = google_compute_address.static_ip.address
}

output "vllm_api_url" {
  description = "vLLM OpenAI-compatible API endpoint"
  value       = "http://${google_compute_address.static_ip.address}:${local.effective_port}/v1"
}

output "model_id" {
  description = "Model being served"
  value       = local.effective_model_id
}

output "machine_type" {
  description = "GCE machine type in use"
  value       = local.effective_machine_type
}

output "multimodal" {
  description = "Whether the served model supports image input (preset only)"
  value       = local.is_multimodal
}

output "is_gated" {
  description = "Whether the model requires a HuggingFace token"
  value       = local.is_gated
}

output "ssh_command" {
  description = "SSH into the VM"
  value       = "gcloud compute ssh vllm-server --zone=${var.zone}"
}

output "destroy_command" {
  description = "Destroy everything and stop paying"
  value       = "cd /mnt/projects/vllm-gpu-deploy && terraform destroy -auto-approve"
}
