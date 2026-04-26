#!/bin/bash

# Stylized messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}WARNING: This will attempt to delete the following:${NC}"
echo -e "  - .venv/"
echo -e "  - logs/"
echo -e "  - checkpoints/"
echo -e "  - datasets/"
echo -e "  - runs/"
echo ""
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Removing environment and data...${NC}"
    
    rm -rf .venv
    rm -rf logs
    rm -rf checkpoints
    rm -rf datasets
    rm -rf runs
    rm -rf .pytest_cache
    rm -rf experiments

    echo -e "${GREEN}Cleanup complete.${NC}"
else
    echo -e "${GREEN}Uninstall cancelled.${NC}"
fi
