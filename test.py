import asyncio
import aiohttp
import pandas as pd
import json

CSV_PATH = "/icebox/data/shares/mh2/shassan6/llama/data/papers_class.csv"
API_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENT_REQUESTS = 20  # Tune based on your server capacity

SYSTEM_PROMPT = (
    "You are a hydrology domain expert. Hydrology studies water in the environment, including rainfall, rivers, lakes, groundwater, floods, droughts, and water resource management. "
    "Classify articles as 'Relevant' only if they primarily focus on hydrology. Articles mainly focusing on forest fires, wildfire management, social media, or general disaster response, "
    "even if natural disasters, are NOT hydrology and must be classified as 'Irrelevant'. If the article does not explicitly focus on water or hydrological processes, reply only with 'Irrelevant'. "
    "Respond with exactly one word: 'Relevant' or 'Irrelevant', and a detailed explanation."
)

# NEW one-shot irrelevant example
DEMO_USER_PROMPT = (
    "CLASSIFY THE FOLLOWING TEXT: Title: Social Media Analysis for Disaster Prevention: Forest Fire in Artenara and Valleseco, Canary Islands.\n\n"
    "Abstract: This manuscript investigates the use of social media, specifically Twitter, during the forest fires in Artenara and Valleseco, Canary Islands, Spain, during summer 2019. "
    "The study analyzes posts from key government accounts and concludes that social media was not used as a preventive element, but as a live-information channel. "
    "It presents recommendations for improving social media use during natural disasters."
)

DEMO_ASSISTANT_PROMPT = (
    "Irrelevant. The article focuses on social media communication during a forest fire, which is not a hydrological process. "
    "It does not involve rainfall, rivers, groundwater, or other elements central to hydrology."
)

MODEL_PATH = "/icebox/data/shares/mh2/shassan6/llama/llama4_scout_fixed/models--meta-llama--Llama-4-Scout-17B-16E/snapshots/14d516bdff6ac06cec40678529222f193386189c"

async def classify(session, title, abstract):
    content = f"CLASSIFY THE FOLLOWING TEXT: Title: {title} Abstract: {abstract}"
    payload = {
        "model": MODEL_PATH,
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": DEMO_USER_PROMPT},
            {"role": "assistant", "content": DEMO_ASSISTANT_PROMPT},
            {"role": "user", "content": content}
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with session.post(API_URL, headers=headers, data=json.dumps(payload)) as resp:
            response = await resp.json()
            result = response["choices"][0]["message"]["content"].strip()
            return result
    except Exception as e:
        print(f"Error classifying paper titled '{title}': {e}")
        return "Error"

async def main():
    df = pd.read_csv(CSV_PATH)

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            classify(session, row['Title'], row['Abstract'])
            for _, row in df.iterrows()
        ]
        classifications = await asyncio.gather(*tasks)

    df["Classification"] = classifications

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved classification results to {CSV_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
