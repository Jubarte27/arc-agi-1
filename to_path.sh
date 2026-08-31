#!/bin/bash

cd "$SCRATCH/arc-agi-1" && {
    PATH="$(pwd)/.ollama/bin:${PATH}"
    LD_LIBRARY_PATH="$(pwd)/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
    OLLAMA_MODELS="$(pwd)/.ollama/models"

    export PATH LD_LIBRARY_PATH OLLAMA_MODELS
}
