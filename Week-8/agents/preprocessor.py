from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class PreProcessor():
    def __init__(self,base_url=None,model_name=DEFAULT_MODEL_NAME,resoning_effort=DEFAULT_REASONING_EFFORT):
        self.base_url = "http://localhost:11434" if "ollama" in model_name else None
        self.model_name = model_name
        self.reasoning_effort=resoning_effort
    
    def messages_for(self,text:str):
        return [{"role":"user","content":text},{"role":"system","content":SYSTEM_PROMPT}]
    
    def preprocess(self,text:str):
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            api_base=self.base_url
        )
        return response.choices[0].message.content
    