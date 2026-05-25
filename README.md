# Vis-Poison

## Local API And Model Config

OpenAI-compatible API credentials and model tags are local machine settings. Create this file when running caption generation or text-embedding KB construction:

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
  "models": {
    "Qwen3.5-397B-A17B": "your-provider-model-tag",
    "text-embedding-3-large": "your-provider-embedding-tag"
  }
}
```

Code uses the model name to look up the provider-specific tag in `models`.
