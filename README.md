# Vis-Poison

## Local API And Model Config

OpenAI-compatible API credentials and model tags are local machine settings. Create this file when running caption generation:

```bash
configs/openai_models.json
```

This file is ignored by git and must not be committed. Required format:

```json
{
  "openai": {
    "base_url": "https://api.example.com/v1",
    "api_key": "YOUR_API_KEY",
    "timeout": 300
  },
  "default_model_name": "Qwen3.5-397B-A17B",
  "models": {
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "GPT-5.4": "gpt-5.4",
    "Qwen3.6-Plus": "qwen3.6-plus",
    "Llama 4 Maverick": "llama-4-maverick",
    "Kimi-K2.6": "kimi-k2.6",
    "Qwen3.5-397B-A17B": "qwen3.5-397b-a17b",
    "Llama 4 Scout": "llama-4-scout",
    "Qwen3.6-35B-A3B": "qwen3.6-35b-a3b",
    "Qwen3.6-27B": "qwen3.6-27b"
  }
}
```