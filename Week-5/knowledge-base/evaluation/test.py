import json 
from pathlib import Path
from pydantic import BaseModel,Field
import sys

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")

class TestQuestion(BaseModel):
  """
  A Test ques
  """
  question:str = Field(description="The question to ask the RAG system")
  keywords:list[str] = Field(description="Keywords that must appear in reterieved context")
  reference_answer:str = Field(description="The reference answer answer for this question")
  category:str = Field(description="Question category (e.g. direct_fact,spanning,temporal)")

def load_tests():
  tests = []
  print("THis is it")
  with open(TEST_FILE,'r',encoding="utf-8") as f:
    for line in f:
      data = json.loads(line.strip())
      tests.append(TestQuestion(**data))
  return tests
