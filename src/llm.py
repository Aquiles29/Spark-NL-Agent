import os
import requests

from langchain_anthropic import ChatAnthropic
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from config import (
    DEFAULT_MODELS,
    DEFAULT_TEMPERATURE,
    Provider
)
import config

GEMINI_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=1 / 13,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

def get_llm(
    provider=Provider.GOOGLE.value,
    model=None,
    temperature=DEFAULT_TEMPERATURE
):    
    if provider == Provider.GOOGLE.value:
        if model is None:
            model = DEFAULT_MODELS[Provider.GOOGLE]
        llm = ChatGoogleGenerativeAI(model=model, temperature=config.DEFAULT_TEMPERATURE, rate_limiter=GEMINI_RATE_LIMITER)
    elif provider == Provider.CLOUDFLARE.value:
        if model is None:
            model = DEFAULT_MODELS[Provider.CLOUDFLARE]
        llm = ChatCloudflareWorkersAI(
            model_name=model,
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN"),
            temperature=temperature
        )
    elif provider == Provider.CLAUDE.value:
        if model is None:
            model = DEFAULT_MODELS[Provider.CLAUDE]
        llm = ChatAnthropic(model=model, temperature=temperature)
    elif provider == Provider.OPENAI.value:
        if model is None:
            model = DEFAULT_MODELS[Provider.OPENAI]
        # Reasoning models (o1, o3-mini) don't support temperature
        if model.startswith("o1") or model.startswith("o3"):
            llm = ChatOpenAI(model=model)
        else:
            llm = ChatOpenAI(model=model, temperature=temperature)
    elif provider == Provider.NVIDIA.value:
        if model is None:
            model = DEFAULT_MODELS[Provider.NVIDIA]
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
    return llm


def get_cloudflare_neuron_pricing(model_name):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        print("Warning: Cloudflare credentials not found. Cannot fetch pricing.")
        return None

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            print(f"Error fetching Cloudflare models: {data.get('errors')}")
            return None

        for model in data.get("result", []):
            if model["name"] == model_name:
                properties = model.get("properties", [])
                price_prop = next((p for p in properties if p["property_id"] == "price"), None)
                
                if price_prop:
                    input_price = 0
                    output_price = 0
                    
                    for p in price_prop["value"]:
                        if p["unit"] == "per M input tokens":
                            input_price = p["price"]
                        elif p["unit"] == "per M output tokens":
                            output_price = p["price"]
                    
                    # 1000 Neurons = $0.011
                    # 1 Neuron = $0.000011
                    # Neurons = Price / 0.000011
                    
                    input_neurons = (input_price / 0.011) * 1000
                    output_neurons = (output_price / 0.011) * 1000
                    
                    return {
                        "input_neurons_per_m": input_neurons,
                        "output_neurons_per_m": output_neurons
                    }
        
        print(f"Model {model_name} not found in Cloudflare catalog.")
        return None

    except Exception as e:
        print(f"Error fetching Cloudflare pricing: {e}")
        return None
