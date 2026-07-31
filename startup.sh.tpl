#!/bin/bash
# Startup script for vLLM on plain Debian 12 + NVIDIA L4 GPU
# Installs: NVIDIA driver, Docker, nvidia-container-toolkit, vLLM container
# Supports any HuggingFace model (gated models need HF_TOKEN)
set -x
exec > >(tee /var/log/vllm-setup.log) 2>&1
echo "=== vLLM Setup started at $(date) ==="
echo "=== Model: ${model_id} ==="
echo "=== Max model len: ${max_model_len} ==="
echo "=== GPU memory utilization: ${gpu_memory_utilization} ==="

# ============================================================
# 1. Install NVIDIA driver from NVIDIA's Debian repo
# ============================================================
echo "=== Installing NVIDIA driver ==="

echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/non-free.list
echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list.d/non-free.list

apt-get update -qq

apt-get install -y -qq linux-headers-$(uname -r) build-essential dkms
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver firmware-misc-nonfree

modprobe nvidia || true

echo "Waiting for nvidia-smi..."
for i in $(seq 1 30); do
  if nvidia-smi > /dev/null 2>&1; then
    echo "nvidia-smi available!"
    break
  fi
  sleep 5
done

if ! nvidia-smi > /dev/null 2>&1; then
  echo "WARNING: nvidia-smi not available yet, may need reboot. Continuing..."
  modprobe nvidia || true
  sleep 10
fi

nvidia-smi || echo "nvidia-smi still failing"
echo "=== NVIDIA driver install done ==="

# ============================================================
# 2. Install Docker
# ============================================================
echo "=== Installing Docker ==="
apt-get install -y -qq ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /tmp/docker.gpg
gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg /tmp/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
echo "=== Docker installed ==="

# ============================================================
# 3. Install nvidia-container-toolkit
# ============================================================
echo "=== Installing nvidia-container-toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo "=== nvidia-container-toolkit installed ==="

# ============================================================
# 4. Start vLLM container
# ============================================================
echo "=== Pulling vLLM image: ${vllm_image} ==="
docker pull ${vllm_image}

echo "=== Starting vLLM container ==="
if [ -n "${hf_token}" ]; then
  docker run -d \
    --name vllm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=4G \
    --restart unless-stopped \
    -p ${vllm_port}:${vllm_port} \
    -e HUGGING_FACE_HUB_TOKEN=*** \
    ${vllm_image} \
    --model ${model_id} \
    --port ${vllm_port} \
    --host 0.0.0.0 \
    --max-model-len ${max_model_len} \
    --gpu-memory-utilization ${gpu_memory_utilization}
else
  docker run -d \
    --name vllm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=4G \
    --restart unless-stopped \
    -p ${vllm_port}:${vllm_port} \
    ${vllm_image} \
    --model ${model_id} \
    --port ${vllm_port} \
    --host 0.0.0.0 \
    --max-model-len ${max_model_len} \
    --gpu-memory-utilization ${gpu_memory_utilization}
fi

echo "=== vLLM container started at $(date) ==="
echo "Check progress: docker logs -f vllm"
