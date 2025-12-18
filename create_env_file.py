"""
Helper script to create your .env file interactively
Run this to set up your API keys easily!
"""

import os

def create_env_file():
    print("=" * 60)
    print("Easy LLM Importer - .env File Setup")
    print("=" * 60)
    print("\nThis script will help you create your .env file.")
    print("Your API keys will be stored securely and NOT committed to git.\n")
    
    # Get API keys from user
    print("Get your keys from:")
    print("  • DeepInfra: https://deepinfra.com/")
    print("  • OpenRouter: https://openrouter.ai/")
    print("  • OpenAI: https://platform.openai.com/\n")
    
    deepinfra_key = input("Enter your DeepInfra API key (or press Enter to skip): ").strip()
    openrouter_key = input("Enter your OpenRouter API key (or press Enter to skip): ").strip()
    openai_key = input("Enter your OpenAI API key (optional, press Enter to skip): ").strip()
    
    # Create .env content
    env_content = "# API Keys for Easy LLM Importer\n"
    env_content += "# Auto-generated - DO NOT commit to git!\n\n"
    
    if deepinfra_key:
        env_content += f"DEEPINFRA_API_KEY={deepinfra_key}\n"
    else:
        env_content += "# DEEPINFRA_API_KEY=your_key_here\n"
    
    if openrouter_key:
        env_content += f"OPENROUTER_API_KEY={openrouter_key}\n"
    else:
        env_content += "# OPENROUTER_API_KEY=your_key_here\n"
    
    if openai_key:
        env_content += f"OPENAI_API_KEY={openai_key}\n"
    else:
        env_content += "# OPENAI_API_KEY=your_key_here\n"
    
    # Write to file
    env_path = ".env"
    
    # Check if file already exists
    if os.path.exists(env_path):
        overwrite = input(f"\n⚠️  {env_path} already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Cancelled. Your existing .env file was not modified.")
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"\n✓ Successfully created {env_path}")
    print("\nYou can now use the Easy LLM Importer!")
    print("\nTest it in Python:")
    print("  from dotenv import load_dotenv")
    print("  load_dotenv()")
    print("  from easy_llm_importer import create_client")
    print("  client = create_client()")
    
    # Verify
    print("\n" + "=" * 60)
    print("Verification:")
    if deepinfra_key:
        print("  ✓ DeepInfra key set (for Llama, Deepseek, Qwen)")
    else:
        print("  ✗ DeepInfra key not set")
    
    if openrouter_key:
        print("  ✓ OpenRouter key set (for Grok, Gemini, Claude)")
    else:
        print("  ✗ OpenRouter key not set")
    
    if openai_key:
        print("  ✓ OpenAI key set (for GPT models)")
    else:
        print("  ○ OpenAI key not set (optional)")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        create_env_file()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nYou can also manually create a .env file with:")
        print("  DEEPINFRA_API_KEY=your_key")
        print("  OPENROUTER_API_KEY=your_key")

