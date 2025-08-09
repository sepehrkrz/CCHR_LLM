 # LLAMA 4 Scout Hydrology Paper Classifier

 This project uses the LLAMA 4 Scout language model to classify scientific papers based on their relevance to hydrology.

 ---

 ## Overview

 - **Model:** LLAMA 4 Scout (17B parameters) served locally with vLLM for efficient inference.
 - **Purpose:** Automatically classify articles as **Relevant** or **Irrelevant** to hydrology.
 - **Classification Criteria:**  
   - *Relevant* if the paper primarily focuses on hydrological topics (rainfall, rivers, groundwater, floods, water resource management, etc.).  
   - *Irrelevant* if focused on other disaster types (forest fires, social media during disasters, etc.) or unrelated topics.

 ---

 ## Setup

 1. Activate the virtual environment:

    ```bash
    conda activate /icebox/data/shares/mh1/llm_cchr/llama4_env
    ```

 2. Start the model server with vLLM:

    ```bash
    vllm serve /icebox/data/shares/mh2/shassan6/llama/Llama-4-Scout-17B-16E/models--meta-llama--Llama-4-Scout-17B-16E/snapshots/14d516bdff6ac06cec40678529222f193386189c \
      --tensor-parallel-size 4 \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.95 \
      --max-num-batched-tokens 32768 \
      --max-num-seqs 32
    ```

 3. Test the server with a sample request (using `curl`):

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d @request.json
    ```

 ---

 ## Classification Script

 The `test.py` script asynchronously sends papers to the LLAMA 4 Scout API for classification.

 - Reads papers from a CSV file (default path: `/icebox/data/shares/mh2/shassan6/llama/data/papers_class.csv`)
 - Sends classification requests concurrently (default max 20 simultaneous requests)
 - Adds a `Classification` column with model outputs (`Relevant` or `Irrelevant` with explanation)
 - Saves updated CSV with classifications

 ### Usage

 ```bash
 python test.py
 ```

 ---

 ## Notes

 - The model uses a custom prompt defining classification rules.
 - Make sure the model server is running before executing the script.
 - Adjust `CONCURRENT_REQUESTS` in `test.py` based on your hardware capability.



