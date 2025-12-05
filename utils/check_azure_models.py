# scripts/check_azure_models.py

"""
Check which models are deployed in Azure OpenAI
"""

from openai import AzureOpenAI

# Setup client
client = AzureOpenAI(
    api_key="xx",
    api_version="2024-10-01-preview",
    azure_endpoint="https://fhgenie-api-fit-ems30127.openai.azure.com/"
)

print("="*80)
print("CHECKING AVAILABLE AZURE OPENAI MODELS")
print("="*80)

# List of models to test
test_models = [
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-35-turbo",
    "gpt-3.5-turbo"
]

available = []
unavailable = []

for model in test_models:
    print(f"\nTesting: {model}...", end=" ")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("✅ AVAILABLE")
        available.append(model)
        
    except Exception as e:
        error_msg = str(e)
        if "DeploymentNotFound" in error_msg:
            print(" NOT DEPLOYED")
        elif "404" in error_msg:
            print(" NOT FOUND")
        else:
            print(f" ERROR: {error_msg[:50]}")
        unavailable.append(model)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ AVAILABLE MODELS ({len(available)}):")
for model in available:
    print(f"   - {model}")

print(f"\n UNAVAILABLE MODELS ({len(unavailable)}):")
for model in unavailable:
    print(f"   - {model}")

print("\n" + "="*80)