from typing import Optional,List
from agents.agent import Agent
from agents.deals import ScrapedDeal,DealSelection,Deal,Opprtunity
from agents.scanner_agent import ScannerAgent
from agents.fronteir_agent import FronteirAgent
from agents.MessagingAgent import MessagingAgent

class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.Green
    Deal_threshold = 50

    def __init__(self,collection):
        self.log("Planning Agent is initializing")
        self.scanner=ScannerAgent()
        self.fronteir = FronteirAgent(collection)
        self.messagener = MessagingAgent()
        self.log("Planning agent is ready")
    
    def run(self,deal:Deal)->Opprtunity:


    



