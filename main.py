from baml_client.sync_client import b
from baml_client.types import Characters
from baml_py import Collector, ClientRegistry
from pyvis.network import Network
import pickle
import litellm
from litellm import get_max_tokens, token_counter, encode, decode, register_model
import os
from pathlib import Path
from typing import List

litellm.register_model({
    "granite4:micro":{
        "max_tokens": 128000,
        "litellm_provider": "ollama"
    }
})

litellm.register_model({
    "gemma3:1b":{
        "max_tokens": 4096,
        "litellm_provider": "ollama"
    }
})


net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
net.barnes_hut()
characters = []

collector = Collector(name="usage")

weight = {
    "ACQUAINTANCE": 0,
    "INNER_CIRCLE": 0.3,
    "OUTER_CIRCLE": 0.2
}

FILE_PATH = Path("./docs/sample.txt")
TITLE = FILE_PATH.name.split(".")[0]
LLM_RESULTS_CACHE = Path(f'./result_cache/{TITLE}.pickle')
GRAPH_PATH = Path(f'./graphs/{TITLE}.html')
TRUNCATE_MARGIN = 0.75
MODEL_TYPE = "CUSTOM"

def get_file_content(file_path: Path) -> str:
    if file_path.exists():
        with file_path.open(encoding='utf-8',mode='r') as f:
            content = f.read()
    return content

def pre_process(input_content: str) -> str:
    # TO DO: Might need when trying to optimize token count.
    return input_content

def calculate_tokens(input_content, model) -> int:
    #calculates total token count given a model
    return token_counter(model= model, messages=[{"content":input_content}])

def input_truncate(content: str, model, token_count) -> str:
    tokens = encode(model=model, text=content)
    truncated_content = decode(tokens=tokens[:int(token_count)], model=model)
    return truncated_content

def get_character_relationship(content: str, llm_client) -> List[Characters]:
    if LLM_RESULTS_CACHE.exists():
        results = pickle.loads(LLM_RESULTS_CACHE.read_bytes())
    else:
        results = b.CharacterRelationships(content, baml_options = {"collector":collector, "client_registry": llm_client})
        with LLM_RESULTS_CACHE.open('wb') as f:
            pickle.dump(results, f)
        print(collector.last.usage)
    return results

def generate_graph_visualization(relationships: List[Characters]):
    for character in relationships:
        src = character.name
        net.add_node(src, src, title=src)
        for relationship in character.relationships:
            dst = relationship.name
            wt = relationship.weight
            net.add_node(dst, dst, title=dst)
            net.add_edge(src, dst, value=weight[wt])

if __name__ == "__main__":
    #LLM Client Instantiation

    llm_client = ClientRegistry()

    llm_client.add_llm_client(
        name="GPT5Nano",
        provider="openai-responses",
        options={
            "model":"gpt-5-nano",
            "temperature":1.0,
            "api_key":os.environ["OPENAI_API_KEY"]
        }
    )

    llm_client.add_llm_client(
        name="granite4",
        provider="openai-generic",
        options={
            "base_url":"http://localhost:11434/v1",
            "model":"granite4:micro"
        }
    )

    llm_client.add_llm_client(
        name="gemma3_1b",
        provider="openai-generic",
        options={
            "base_url":"http://localhost:11434/v1",
            "model":"gemma3:1b"
        }
    )

    if MODEL_TYPE == "CUSTOM":
        model = "gemma3:1b"
        llm_client.set_primary("gemma3_1b")
    else:
        model = "gpt-5-nano"
        llm_client.set_primary("GPT4Nano")
    
    model_max_tokens = get_max_tokens(model)
    
    #input
    content = get_file_content(file_path=FILE_PATH)
    input_token_count=calculate_tokens(content, model)

    if input_token_count > (model_max_tokens*TRUNCATE_MARGIN):
        print(f"Input token count {input_token_count} exceeds {model}'s max token limit of {model_max_tokens}")
        print(f"Truncating input to {TRUNCATE_MARGIN*model_max_tokens}")
        content = input_truncate(content, token_count = TRUNCATE_MARGIN*model_max_tokens, model=model)

    
    relationships = get_character_relationship(content, llm_client)
    generate_graph_visualization(relationships)
    net.show(str(GRAPH_PATH.absolute()), notebook=False)