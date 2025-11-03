from baml_client.sync_client import b
from baml_client.types import Characters
from baml_py import Collector
from pyvis.network import Network
import pickle
from litellm import get_max_tokens, token_counter, encode, decode
import os
from pathlib import Path
from typing import List

net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
net.barnes_hut()
characters = []

collector = Collector(name="usage")

weight = {
    "ACQUAINTANCE": 0,
    "INNER_CIRCLE": 0.3,
    "OUTER_CIRCLE": 0.2
}

FILE_PATH = Path("./docs/little_woman.txt")
TITLE = FILE_PATH.name.split(".")[0]
LLM_RESULTS_CACHE = Path(f'./result_cache/{TITLE}.pickle')
GRAPH_PATH = Path(f'./graphs/{TITLE}.html')
TRUNCATE_MARGIN = 0.75

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

def chunk_input(content: str, model: str, chunk_size: int) -> List[str]:
    tokens = encode(model=model, text=content)
    chunked_content = [decode(tokens=tokens[i: i+chunk_size], model=model) for i in range(0, len(tokens), chunk_size)]
    return chunked_content

def get_character_relationship(content: str) -> List[Characters]:
    if LLM_RESULTS_CACHE.exists():
        results = pickle.loads(LLM_RESULTS_CACHE.read_bytes())
    else:
        input_token_count=calculate_tokens(content, model)

        #Check if content satisfies input token constraint
        if input_token_count > (model_max_tokens*TRUNCATE_MARGIN):
            max_input_tokens = TRUNCATE_MARGIN*model_max_tokens
            chunk_size = int(max_input_tokens/2)
            print(f"Input token count {input_token_count} exceeds {model}'s max token limit of {model_max_tokens}")
            print(f"chunking input into {chunk_size}")
            #content = input_truncate(content, token_count = TRUNCATE_MARGIN*model_max_tokens, model=model)
            content = chunk_input(content, model=model, chunk_size=chunk_size)
        else:
            content = [content]
        
        prev_output = []
        for chunk in content:
            results = b.CharacterRelationships(txt=chunk, prev_output=prev_output, baml_options = {"collector":collector})
            prev_output = results
            print(collector.last.usage)
        
        with LLM_RESULTS_CACHE.open('wb') as f:
            pickle.dump(results, f)
        print(collector.usage)
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
    model = "gpt-5-nano"
    
    model_max_tokens = get_max_tokens(model)
    
    #input
    content = get_file_content(file_path=FILE_PATH)
    
    relationships = get_character_relationship(content)
    generate_graph_visualization(relationships)
    net.show(str(GRAPH_PATH.absolute()), notebook=False)