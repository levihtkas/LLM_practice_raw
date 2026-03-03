from pydantic import BaseModel
from datasets import Dataset,DatasetDict,load_dataset
from typing import Optional



PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

class Item(BaseModel):
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self,text):
        self.prompt = f"{QUESTION} \n\n {text} \n\n {PREFIX} {round(self.price)}"
    
    def test_prompt(self)->str:
        return self.prompt.split(PREFIX)[0]+PREFIX
    # repersentation of the object when it gets printed
    def __repr__(self) -> str:
        return f"<{self.title}> = ${self.price}"
    
    @staticmethod
    def push_to_hub(dataset_name:str,train:list,val:list,test:list):
        """
        Docstring for push_to_hub : Pushes to HF
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)
    
    @classmethod
    def from_hub(cls,dataset_name):
        ds = load_dataset(dataset_name)

        return (
            [cls.model_validate(row) for row in ds['train']],
            [cls.model_validate(row) for row in ds['validation']],
            [cls.model_validate(row) for row in ds['test']]
        )
    
    def countTokens(self,tokenizer):
        return len(tokenizer.encode(self.summary,add_special_tokens=False))
    
    def make_prompts(self,tokenizer,max_token,do_round):
        tokens = tokenizer.encode(self.summary,add_special_tokens=False)
        if len(tokens)>max_token:
            summary = tokenizer.decode(tokens[:max_token])
        else:
            summary=self.summary
        self.prompt = f"{QUESTION} \n {summary} \n {PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer):
        """Count tokens in the prompt"""
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list, val: list, test: list
    ):
        """Push Item lists to HuggingFace Hub in prompt-completion format for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)
    



