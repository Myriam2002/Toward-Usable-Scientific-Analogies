# Easy LLM Importer 🚀

A unified, intelligent LLM client that automatically routes model calls to the appropriate provider (OpenAI, OpenRouter, or DeepInfra) based on model name. Built for Master's thesis research on scientific analogies.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
│              (Thesis Analysis Pipeline)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Easy LLM Importer                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          ModelRegistry (Router)                     │   │
│  │  Maps model names → Provider + API identifier       │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                       │
│        ┌────────────┼────────────┐                         │
│        ▼            ▼            ▼                          │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ OpenAI  │ │OpenRouter│ │DeepInfra │                    │
│  │ Client  │ │  Client  │ │  Client  │                    │
│  └─────────┘ └──────────┘ └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │   DSPy Layer   │  (Optional)
            │  Signatures &  │
            │   Reasoning    │
            └────────────────┘
```

## Key Features

✅ **Automatic Provider Routing** - Just specify model name, routing is handled automatically  
✅ **Unified API** - Same interface for all providers  
✅ **15+ Models Supported** - GPT, Claude, Gemini, Llama, Deepseek, Qwen, Grok  
✅ **DSPy Integration** - Use structured signatures instead of raw prompts  
✅ **Streaming Support** - Real-time response streaming  
✅ **Error Handling** - Built-in retry logic and fallbacks  
✅ **Environment Variables** - Secure API key management  

## Supported Models

| Model Name | Provider | Use Case |
|------------|----------|----------|
| `llama-3.3-70b` | DeepInfra | General reasoning, cost-effective |
| `llama-3.1-405b` | DeepInfra | Complex reasoning, large context |
| `deepseek-r1` | DeepInfra | Reasoning-focused tasks |
| `deepseek-v3.1` | DeepInfra | Latest Deepseek capabilities |
| `claude-3.5-haiku` | OpenRouter | Fast, efficient Claude |
| `gemini-2.5-flash` | OpenRouter | Google's fast model |
| `gemini-2.5-pro` | OpenRouter | Google's advanced model |
| `grok-4-fast` | OpenRouter | X.AI's fast model |
| `qwen3-coder-480b-a35b` | DeepInfra | Code generation |

*See `easy_llm_importer.py` for complete list*

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `openai>=1.12.0` - For unified API access
- `dspy-ai>=2.4.0` - For advanced prompt engineering (optional)
- `pandas` - For data processing

### 2. Set Up API Keys

**Option A: Environment Variables (Recommended)**

Create a `.env` file in your project root:

```bash
# OpenRouter (for Grok, Gemini, Claude)
OPENROUTER_API_KEY=your_openrouter_key_here

# DeepInfra (for Llama, Deepseek, Qwen)
DEEPINFRA_API_KEY=your_deepinfra_key_here

# OpenAI (if using GPT-5 models)
OPENAI_API_KEY=your_openai_key_here

# Optional: For OpenRouter rankings
SITE_URL=https://github.com/yourusername/thesis
SITE_NAME=Master Thesis Research
```

**Option B: Pass Directly to Client**

```python
from easy_llm_importer import create_client

client = create_client(
    openrouter_api_key="your_key",
    deepinfra_api_key="your_key"
)
```

### 3. Get API Keys

- **OpenRouter**: https://openrouter.ai/ (supports Grok, Claude, Gemini)
- **DeepInfra**: https://deepinfra.com/ (supports Llama, Deepseek, Qwen)
- **OpenAI**: https://platform.openai.com/ (for GPT models)

## Usage Examples

### Basic Usage

```python
from easy_llm_importer import create_client

# Create client
client = create_client()

# Simple chat
response = client.chat(
    model_name="llama-3.3-70b",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement"}
    ],
    temperature=0.7
)
print(response)
```

### Quick One-Off Call

```python
from easy_llm_importer import quick_chat

response = quick_chat(
    model_name="deepseek-r1",
    prompt="What is a scientific analogy?",
    temperature=0.7
)
print(response)
```

### Streaming Responses

```python
for chunk in client.stream_chat(
    model_name="claude-3.5-haiku",
    messages=[{"role": "user", "content": "Tell me a story"}]
):
    print(chunk, end="", flush=True)
```

### Multi-Model Comparison

```python
models = ["llama-3.3-70b", "deepseek-r1", "claude-3.5-haiku"]
prompt = "Map neuron properties to traffic systems"

for model in models:
    print(f"\n{'='*60}\n{model}\n{'='*60}")
    response = client.chat(
        model_name=model,
        messages=[{"role": "user", "content": prompt}]
    )
    print(response)
```

### DSPy Integration (Advanced)

```python
import dspy
from easy_llm_importer import DSPyAdapter

# Configure DSPy with your model
adapter = DSPyAdapter(client, model_name="deepseek-r1")
lm = adapter.get_dspy_lm()
dspy.settings.configure(lm=lm)

# Define a signature
class AnalogyMapping(dspy.Signature):
    """Map properties from source to target concept"""
    source_concept = dspy.InputField()
    target_domain = dspy.InputField()
    mapping = dspy.OutputField()

# Use Chain of Thought reasoning
mapper = dspy.ChainOfThought(AnalogyMapping)
result = mapper(
    source_concept="Solar system",
    target_domain="Atomic structure"
)
print(result.mapping)
```

### Batch Processing Thesis Data

```python
import pandas as pd

# Load SCAR dataset
df = pd.read_json('../../data/SCAR.json', lines=True)

results = []
for idx, row in df.iterrows():
    response = client.chat(
        model_name="llama-3.3-70b",
        messages=[{
            "role": "user",
            "content": f"Analyze this analogy: {row['source']} → {row['target']}"
        }],
        temperature=0.5
    )
    results.append({'index': idx, 'response': response})

df_results = pd.DataFrame(results)
```

## System Design Principles

### 1. **Separation of Concerns**
- `ModelRegistry`: Model-to-provider mapping
- `LLMClient`: API client management
- `DSPyAdapter`: DSPy integration layer

### 2. **Lazy Initialization**
- Clients only initialized when needed
- Reduces overhead for unused providers

### 3. **Extensibility**
- Add new models by updating `ModelRegistry.MODELS`
- Add new providers by implementing `_chat_<provider>` method

### 4. **Error Handling**
- Clear error messages for missing API keys
- Model validation before API calls
- Easy to implement fallback strategies

## Adding New Models

```python
# In easy_llm_importer.py, add to ModelRegistry.MODELS:

"new-model-name": ModelConfig(
    name="new-model-name",
    provider=Provider.DEEPINFRA,  # or OPENROUTER, OPENAI
    api_name="provider/model-identifier"
)
```

## Cost Optimization Tips

1. **Use appropriate models for tasks**:
   - Quick queries: `llama-3.3-70b`, `claude-3.5-haiku`
   - Complex reasoning: `deepseek-r1`, `llama-3.1-405b`
   - Code generation: `qwen3-coder-480b-a35b`

2. **Set max_tokens** to avoid excessive generation:
   ```python
   client.chat(model_name="...", messages=[...], max_tokens=500)
   ```

3. **Use streaming** for long responses to enable early stopping

4. **Implement caching** for repeated queries

## Troubleshooting

### "Model not found" Error
```python
# Check available models
from easy_llm_importer import list_available_models
print(list_available_models())
```

### "API key not provided" Error
```python
# Verify environment variables are set
import os
print(os.getenv("OPENROUTER_API_KEY"))  # Should not be None
```

### Import Error for DSPy
```bash
pip install dspy-ai
```

## References

- [OpenRouter Documentation](https://openrouter.ai/docs/quickstart)
- [DeepInfra Documentation](https://deepinfra.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs/models)
- [DSPy Documentation](https://dspy.ai/learn/programming/)

## Project Structure

```
stage1_analysis/mapping_generation/
├── easy_llm_importer.py    # Main unified client
├── example_usage.ipynb     # Complete usage examples
├── mapping_analysis.ipynb  # Your thesis analysis
└── README.md              # This file
```

## Future Enhancements

- [ ] Add caching layer for repeated queries
- [ ] Implement rate limiting and retry logic
- [ ] Add cost tracking per model
- [ ] Support for image/multimodal models
- [ ] Async/parallel request handling
- [ ] Model performance benchmarking

## License

For Master's Thesis Research - Toward Usable Scientific Analogies

---

**Built with** ❤️ **for research in scientific analogies**

