from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="xx",
    api_version="2024-10-01-preview",
    azure_endpoint="https://fhgenie-api-fit-ems30127.openai.azure.com/"
)

# Use gpt-4o (the best model available)
response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=[
        {"role": "system", "content": "You are an expert in ODRL policies."},
        {"role": "user", "content": "Explain what ODRL permissions are."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)