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

# 1. Create Directories (Safe creation)
log_info "Setting up directory structure..."
mkdir -p datasets checkpoints logs
log_success "Directories 'datasets/', 'checkpoints/', 'logs/' are ready."

# 2. Virtual Environment Setup
if command -v uv &> /dev/null; then
    log_info "Tool 'uv' detected. Using it for faster setup."
    uv venv
    source .venv/bin/activate
    log_info "Installing dependencies with uv..."
    uv pip install -e .
else
    log_info "'uv' not found. Using standard python venv."
    python3 -m venv .venv
    source .venv/bin/activate
    log_info "Upgrading pip..."
    pip install --upgrade pip
    log_info "Installing dependencies with pip..."
    pip install -e .
fi

log_success "Environment setup complete! Run 'source .venv/bin/activate' to start."
