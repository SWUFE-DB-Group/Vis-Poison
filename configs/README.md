# Configs

This folder stores committed task configs.

The file below is required locally but must not be committed:

```text
configs/openai_models.json
```

## `openai_models.json` Format

Required structure:

```json
{
  "openai": {
    "base_url": "https://your-api-endpoint/v1",
    "api_key": "your-api-key",
    "timeout": 300
  },
  "default_model_name": "GPT-5.4",
  "models": {
    "GPT-5.4": "your-provider-model-tag",
    "Claude Sonnet 4.6": "your-provider-model-tag",
    "Qwen3.6-Plus": "your-provider-model-tag",
    "Kimi-K2.6": "your-provider-model-tag",
    "Llama 4 Maverick": "your-provider-model-tag",
    "Qwen3.5-397B-A17B": "your-provider-model-tag",
    "Llama 4 Scout": "your-provider-model-tag",
    "Qwen3.6-35B-A3B": "your-provider-model-tag",
    "Qwen3.6-27B": "your-provider-model-tag",
    "text-embedding-3-large": "your-provider-model-tag"
  }
}
```

## Field Notes

- `openai.base_url`
  OpenAI-compatible API endpoint.

- `openai.api_key`
  API key for that endpoint.

- `openai.timeout`
  Optional request timeout in seconds.

- `default_model_name`
  Optional default display name used by scripts that allow omitting `--model`.

- `models`
  Mapping from the display names used in this repository to provider-specific model tags.

The `models` mapping does not need to include every entry above unless you plan to run the corresponding scripts.
