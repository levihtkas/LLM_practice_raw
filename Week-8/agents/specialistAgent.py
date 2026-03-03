import modal
from agents.agent import Agent
class SpecialistAgent(Agent):

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Specialist Agent is initializing")
        Pricer = modal.Cls.from_name("pricer-service","Pricer")
        self.pricer = Pricer()

    def price(self,description:str)->float:
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result

