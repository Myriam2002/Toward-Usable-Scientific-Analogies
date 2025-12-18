# Setting Up Your .env File

## Step 1: Create the .env file

In your project root folder (`Toward-Usable-Scientific-Analogies/`), create a new file called `.env` (just `.env` with no extension).

**Windows users:** Create a text file and rename it to `.env` (you may need to enable "File name extensions" in File Explorer to see/remove the .txt extension)

## Step 2: Add Your API Keys

Open the `.env` file and paste this template:

```
# API Keys for Easy LLM Importer

# For Llama, Deepseek, Qwen models
DEEPINFRA_API_KEY=your_actual_deepinfra_key_here

# For Grok, Gemini, Claude, GPT-OSS models  
OPENROUTER_API_KEY=your_actual_openrouter_key_here

# For GPT-5 models (optional)
OPENAI_API_KEY=your_actual_openai_key_here
```

**Replace the values with your actual API keys!**

## Step 3: Get Your API Keys

### DeepInfra (Required for: llama-3.3-70b, deepseek-r1, qwen3-coder)

1. Go to https://deepinfra.com/
2. Sign up or log in
3. Click on your profile → "Dashboard"
4. Find "API Keys" section
5. Copy your key
6. Paste it in `.env` after `DEEPINFRA_API_KEY=`

### OpenRouter (Required for: grok-4-fast, gemini-2.5-flash, claude-3.5-haiku)

1. Go to https://openrouter.ai/
2. Sign up or log in
3. Click "Keys" in the menu
4. Click "Create Key"
5. Copy your key
6. Paste it in `.env` after `OPENROUTER_API_KEY=`

## Step 4: Verify Your Setup

In your Jupyter notebook, run:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Check if keys are loaded
print(f"DeepInfra: {os.getenv('DEEPINFRA_API_KEY') is not None}")
print(f"OpenRouter: {os.getenv('OPENROUTER_API_KEY') is not None}")
```

You should see `True` for the keys you've set.

## Step 5: Use It!

Now you can simply:

```python
from easy_llm_importer import create_client

client = create_client()  # Keys automatically loaded from .env!

response = client.chat(
    model_name="llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```

## Security Notes

✓ The `.env` file is already in `.gitignore` - it won't be committed to git  
✓ Never share your `.env` file or commit it to version control  
✓ Each team member should have their own `.env` file  

## Troubleshooting

**Problem: "No such file or directory: .env"**
- Make sure you created `.env` in the root folder (not in a subfolder)
- File should be at: `Toward-Usable-Scientific-Analogies/.env`

**Problem: Still getting "API key not provided"**
- Restart your Jupyter kernel after creating `.env`
- Make sure there are no spaces around the `=` sign
- Make sure you replaced `your_actual_key_here` with your real key

**Problem: Keys show as `None`**
- Check the `.env` file location (should be in project root)
- Make sure the file is named exactly `.env` (not `.env.txt`)
- Try absolute path: `load_dotenv('/full/path/to/your/.env')`

## Example .env File (with fake keys)

```
DEEPINFRA_API_KEY=sk-1a2b3c4d5e6f7g8h9i0j
OPENROUTER_API_KEY=sk-or-v1-abcdef123456
OPENAI_API_KEY=sk-proj-xyz789
```

Replace with your actual keys!

