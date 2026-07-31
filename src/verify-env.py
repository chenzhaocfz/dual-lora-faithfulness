import os
import torch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def verify():
    print("--- Hardware Verification ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA Available: Yes")
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {vram_gb:.2f} GB")
    else:
        print("CUDA Available: NO (Warning: Fine-tuning requires CUDA)")

    print("\n--- DeepSeek API Verification ---")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not found in .env file.")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "The area of Carlos's rectangular living room is 630 square feet. If the length of his room is 7 yards, what is the perimeter of the room in feet?"}],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )
        print("API Connection Successful!")
        print(f"Sample thinking: {response.choices[0].message.reasoning_content}")
        print(f"Sample response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"API Connection Failed: {e}")

if __name__ == "__main__":
    verify()