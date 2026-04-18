#!/bin/bash
set -e

# Stylized messages
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Install Hugging Face CLI if not already installed
if ! command -v hf &> /dev/null; then
    log_info "Installing Hugging Face CLI..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="${HOME}/.local/bin:${PATH}"
    log_success "Hugging Face CLI installed"
else
    log_info "Hugging Face CLI already installed"
fi

DATASETS=("KGraph/FB15k-237:fb15k-237" "VLyb/WN18RR:wn18rr")

for entry in "${DATASETS[@]}"; do
    REPO="${entry%%:*}"
    TARGET_NAME="${entry##*:}"
    
    TARGET_DIR="datasets/${TARGET_NAME}"
    HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
    # Format KGraph/FB15k-237 -> datasets--KGraph--FB15k-237
    REPO_SANTIZED=$(echo "$REPO" | sed 's/\//--/g')
    DATASET_BASE="${HF_CACHE_DIR}/datasets--${REPO_SANTIZED}/snapshots"

    log_info "Starting data ingestion for ${TARGET_NAME} (${REPO})..."

    log_info "Downloading ${REPO} dataset..."
    hf download "${REPO}" --repo-type=dataset

    if [ -d "$DATASET_BASE" ]; then
        DATASET_CACHE=$(find "$DATASET_BASE" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -z "$DATASET_CACHE" ]; then
            log_error "No snapshot directory found in $DATASET_BASE"
            exit 1
        fi
        log_success "Dataset found at: $DATASET_CACHE"
    else
        log_error "Failed to download dataset to $DATASET_BASE"
        exit 1
    fi

    mkdir -p "$(dirname "$TARGET_DIR")"
    if [ -d "$TARGET_DIR" ] || [ -L "$TARGET_DIR" ]; then
        log_info "Target directory already exists, removing it..."
        rm -rf "$TARGET_DIR"
    fi

    log_info "Creating symlink from cache to datasets folder..."
    ln -s "$DATASET_CACHE" "$TARGET_DIR"
    log_success "Symlink created: $TARGET_DIR -> $DATASET_CACHE"

done

log_success "Setup complete! Datasets are ready."
