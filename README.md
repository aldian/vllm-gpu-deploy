# vLLM GPU Deploy — Self-Host Any Open-Weight LLM

Deploy any HuggingFace model on GCE GPU infrastructure with Terraform + vLLM.
One command to deploy, one command to destroy. OpenAI-compatible API out of the box.

## Why?

API providers sell inference below cost (see analysis below). Owning your inference layer means you're never at the mercy of a pricing page. This repo makes it trivial to spin up a GPU VM, serve any open-weight model, test it, and tear it down when you're done.

## Quick start

```bash
git clone https://github.com/aldian/vllm-gpu-deploy.git
cd vllm-gpu-deploy

# For gated models (Gemma, Llama) — put your HF token in .env
echo 'HF_TOKEN=*** > .env

# Deploy default model (Qwen 2.5 3B — no token needed)
terraform init
export $(grep -v '^#' .env | xargs)
export TF_VAR_hf_token="$HF_TOKEN"
terraform apply -auto-approve

# Get your API URL
terraform output vllm_api_url

# Test it
python3 test_vllm.py $(terraform output -raw vllm_api_url)

# Destroy when done
terraform destroy -auto-approve
```

## Supported models

### Small/medium models — single GPU (affordable)

Pick a preset and Terraform selects the right GPU, context length, and memory tuning:

| Preset | Model | Params | Gated? | Multimodal? | GPU(s) | Cost/hr |
|---|---|---|---|---|---|---|
| `tinyllama-1.1b-chat` | TinyLlama 1.1B | 1.1B | No | No | 1x L4 | ~$0.70 |
| `llama-3.2-1b-instruct` | Llama 3.2 1B | 1B | **Yes** | No | 1x L4 | ~$0.70 |
| `llama-3.2-3b-instruct` | Llama 3.2 3B | 3B | **Yes** | No | 1x L4 | ~$0.70 |
| `qwen2.5-3b-instruct` | Qwen 2.5 3B | 3B | No | No | 1x L4 | ~$0.70 |
| `gemma-2-2b-it` | Gemma 2 2B | 2B | **Yes** | No | 1x L4 | ~$0.70 |
| `phi-3.5-mini-instruct` | Phi 3.5 Mini | 3.8B | No | No | 1x L4 | ~$0.70 |
| `gemma-3-4b-it` | Gemma 3 4B | 4B | **Yes** | **Text + Image** | 1x L4 | ~$0.70 |
| `qwen2.5-7b-instruct` | Qwen 2.5 7B | 7B | No | No | 1x L4 | ~$0.70 |
| `mistral-7b-instruct` | Mistral 7B v0.3 | 7B | No | No | 1x L4 | ~$0.70 |
| `gemma-2-9b-it` | Gemma 2 9B | 9B | **Yes** | No | 1x L4 | ~$0.70 |
| `gemma-4-e2b-it` | Gemma 4 E2B | 2.3B eff | **Yes** | **Text+Img+Audio** | 1x L4 | ~$0.70 |
| `gemma-4-e4b-it` | Gemma 4 E4B | ~8B | **Yes** | **Text+Img+Audio** | 1x L4 | ~$0.70 |
| `qwen2.5-14b-instruct` | Qwen 2.5 14B | 14B | No | No | 1x L4 (tight) | ~$0.70 |
| `gemma-3-12b-it` | Gemma 3 12B | 12B | **Yes** | **Text + Image** | 1x L4 | ~$0.70 |
| `gpt-oss-20b` | OpenAI GPT-OSS 20B | 20B (3.6B active) | No | No | 1x L4 | ~$0.70 |
| `gemma-4-26b-moe-it` | Gemma 4 26B MoE | 26B (4B active) | **Yes** | **Text+Img+Audio** | 1x L4 | ~$0.70 |

### Large models — multi-GPU (expensive)

These models require tensor parallelism across multiple GPUs. Costs are **$10-30+/hr**.

| Preset | Model | Params | Gated? | Multimodal? | GPU(s) | Cost/hr |
|---|---|---|---|---|---|---|
| `mixtral-8x7b-instruct` | Mixtral 8x7B | 47B (MoE) | No | No | 2x A100 80GB | ~$12/hr |
| `qwen2.5-72b-instruct` | Qwen 2.5 72B | 72B | No | No | 2x A100 80GB | ~$12/hr |
| `gemma-4-31b-it` | Gemma 4 31B Dense | 31B | **Yes** | **Text+Img+Audio** | 2x A100 80GB | ~$12/hr |
| `mixtral-8x22b-instruct` | Mixtral 8x22B | 141B (MoE) | No | No | 4x A100 80GB | ~$24/hr |
| `deepseek-v3` | DeepSeek V3 | 671B (MoE) | No | No | 8x H100 80GB | ~$30/hr |
| `deepseek-r1` | DeepSeek R1 | 671B (MoE) | No | No | 8x H100 80GB | ~$30/hr |
| `llama-3.1-405b-instruct` | Llama 3.1 405B | 405B | **Yes** | No | 8x H100 80GB | ~$30/hr |
| `gpt-oss-120b` | OpenAI GPT-OSS 120B | 120B (5.1B active) | No | No | 8x H100 80GB | ~$30/hr |
| `kimi-k3` | Kimi K3 | 2.8T (MoE) | No | **Text + Image** | 8x H100 80GB | ~$30/hr |

> ⚠️ **Cost warning**: These models are not for experimentation. A single boot cycle (download + load) can take **30-60+ minutes** and cost **$15-30**. Only deploy if you have a production use case and budget.

> ⚠️ **Kimi K3 / DeepSeek V3 note**: These are massive MoE models. Even with 8x H100 (640GB VRAM), they require quantization (MXFP4/FP8) to fit. The preset includes the right quantization flag automatically. Verify latest vLLM support before deploying: https://docs.vllm.ai/en/stable/models/supported_models.html

### Any model (bring your own)

Pass any HuggingFace model ID directly:

```bash
# Serve any model — defaults: g2-standard-4, 4096 context, 0.85 GPU util
terraform apply -var="model_id=Qwen/Qwen2.5-14B-Instruct"

# With overrides
terraform apply -var="model_id=meta-llama/Llama-3.1-8B-Instruct" \
                -var="machine_type=g2-standard-8" \
                -var="max_model_len=16384"
```

Works with any model vLLM supports. See: https://docs.vllm.ai/en/stable/models/supported_models.html

## Usage examples

### Default (no token needed)

```bash
terraform apply -auto-approve
# Serves Qwen/Qwen2.5-3B-Instruct on g2-standard-4
```

### Gated model (Gemma/Llama)

```bash
export $(grep -v '^#' .env | xargs)
export TF_VAR_hf_token="$HF_TOKEN"
terraform apply -auto-approve -var="model_preset=gemma-3-4b-it"
```

### Custom model with overrides

```bash
terraform apply -auto-approve \
  -var="model_id=Qwen/Qwen2.5-14B-Instruct" \
  -var="machine_type=g2-standard-8" \
  -var="max_model_len=8192" \
  -var="gpu_memory_utilization=0.90"
```

### Large model with multi-GPU (custom)

```bash
terraform apply -auto-approve \
  -var="model_id=Qwen/QwQ-32B-Preview" \
  -var="machine_type=a2-ultragpu-2g" \
  -var="tensor_parallel_size=2" \
  -var="disk_size_gb=300"
```

### Different zone (GPU capacity issues)

```bash
terraform apply -auto-approve -var="region=us-west1" -var="zone=us-west1-a"
```

## Architecture

```
Internet → GCE VM (g2-standard-N + NVIDIA L4 GPU)
                └── Docker container (vllm/vllm-openai:latest)
                     └── Any HuggingFace model
                          └── OpenAI-compatible API on :8000
```

## Cost estimate

### Single-GPU (L4)

| Machine type | GPUs | VRAM | CPU | RAM | Est. cost/hr |
|---|---|---|---|---|---|
| g2-standard-4 | 1x L4 | 24GB | 4 vCPU | 16GB | ~$0.70 |
| g2-standard-8 | 1x L4 | 24GB | 8 vCPU | 32GB | ~$1.10 |
| g2-standard-16 | 1x L4 | 24GB | 16 vCPU | 64GB | ~$1.80 |
| g2-standard-24 | 2x L4 | 48GB | 24 vCPU | 96GB | ~$2.50 |

### Multi-GPU (A100/H100 — for large models)

| Machine type | GPUs | VRAM | Est. cost/hr |
|---|---|---|---|
| a2-ultragpu-2g | 2x A100 80GB | 160GB | ~$12/hr |
| a2-ultragpu-4g | 4x A100 80GB | 320GB | ~$24/hr |
| a3-highgpu-8g | 8x H100 80GB | 640GB | ~$30/hr |

Costs are approximate (us-central1, sustained-use discount applied). Always destroy after testing.

## Prerequisites

### 1. Tools

```bash
# Terraform >= 1.0 — https://developer.hashicorp.com/terraform/install
terraform version

# gcloud CLI — https://cloud.google.com/sdk/docs/install
gcloud version
```

### 2. GCP project setup

```bash
gcloud projects create YOUR_PROJECT_ID
gcloud config set project YOUR_PROJECT_ID

# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Authenticate Terraform
gcloud auth application-default login

# Verify GPU quota (need at least 1 L4 GPU)
gcloud compute regions describe us-central1 \
  --format="(quotas.filter(name:INSTANCES_WITH_L4_GPUS).limit)"
# If 0, request quota: https://console.cloud.google.com/iam-admin/quotas
```

For large models, you also need A100/H100 GPU quota:

```bash
# Check A100 quota
gcloud compute regions describe us-central1 \
  --format="(quotas.filter(name:A2_CPUS).limit)"

# Check H100 quota
gcloud compute regions describe us-central1 \
  --format="(quotas.filter(name:A3_CPUS).limit)"
```

### 3. Set your GCP project ID

Edit `variables.tf` and change the `project_id` default, or pass it at deploy time:

```bash
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### 4. HuggingFace token (only for gated models)

Gated models (Gemma, Llama) require a token. Open models (Qwen, Mistral, Phi, TinyLlama) do not.

```bash
# Create token: https://huggingface.co/settings/tokens
# Accept license on the model page (e.g. https://huggingface.co/google/gemma-3-4b-it)
echo 'HF_TOKEN=*** > .env
```

## After deploy: wait for boot

The startup script needs ~5-10 minutes to:
1. Install NVIDIA driver (DKMS build from source)
2. Install Docker + nvidia-container-toolkit
3. Pull vLLM Docker image (~5GB)
4. Download model weights
5. Compile CUDA graphs and start serving

Large models (DeepSeek V3, Kimi K3) can take **30-60+ minutes** just for weight download + load.

### Monitor progress

```bash
IP=$(terraform output -raw vm_external_ip)
ZONE=$(terraform output -raw ssh_command | grep -oP 'zone=\K\S+')

# Watch startup log
gcloud compute ssh vllm-server --zone=$ZONE --command="tail -f /var/log/vllm-setup.log"

# Watch vLLM container logs
gcloud compute ssh vllm-server --zone=$ZONE --command="sudo docker logs -f vllm"

# Check GPU
gcloud compute ssh vllm-server --zone=$ZONE --command="nvidia-smi"
```

### Known issue: Docker install on first boot

The Docker GPG key import sometimes fails on first boot. If the startup log shows `docker: command not found`:

```bash
gcloud compute ssh vllm-server --zone=$ZONE --command='
  sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /tmp/docker.gpg &&
  sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg /tmp/docker.gpg &&
  sudo chmod a+r /etc/apt/keyrings/docker.gpg &&
  sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io &&
  sudo systemctl start docker
'
```

Then restart vLLM manually — see the Docker run command from `terraform output`.

## Test

```bash
IP=$(terraform output -raw vm_external_ip)

# Run the automated test suite
python3 test_vllm.py http://$IP:8000/v1

# Or test manually
curl http://$IP:8000/v1/models

curl http://$IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID_FROM_OUTPUT",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Test multimodal (Gemma 3 models only)

```bash
IMG_B64=$(base64 -w0 /tmp/test.jpg)
curl http://$IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"google/gemma-3-4b-it\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"What do you see in this image?\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}}
      ]
    }],
    \"max_tokens\": 100
  }"
```

## Use from any OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_VM_IP:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="MODEL_ID_FROM_OUTPUT",
    messages=[{"role": "user", "content": "What is HIPAA?"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Destroy (stop paying)

```bash
terraform destroy -auto-approve
```

Removes the VM, static IP, firewall rules, and service account. Everything is gone.

## Files

| File | Description |
|---|---|
| `main.tf` | Terraform: GCE GPU instance, firewall, static IP, service account |
| `variables.tf` | 19 model presets + configurable variables (model, zone, GPU, memory, tensor parallelism) |
| `startup.sh.tpl` | Startup script: NVIDIA driver, Docker, vLLM container |
| `test_vllm.py` | Test suite: health check, chat, streaming, throughput |
| `.env` | HuggingFace token for gated models (gitignored) |
| `.gitignore` | Ignores tfstate, .terraform, .env |

## License

MIT
