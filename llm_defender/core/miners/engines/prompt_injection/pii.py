from os import path, makedirs
import bittensor as bt
from llm_defender.base.engine import BaseEngine

class PIIEngine(BaseEngine):


    def __init__(self, prompt: str = None, name = 'engine:pii'):
        super().__init__(name=name)
        self.prompt = prompt

    def _calculate_confidence(self):
        if self.output['outcome'] == 'PIIDetected':
            return 1.0
        elif self.output['outcome'] == 'NoPIIDetected':
            return 0.0
        return 0.5 

    def _populate_data(self, results):
        if results:
            # PII detected (FILL THIS OUT LATER)
            if results:
                return {'outcome':'PIIDetected'}
            # No PII Detected (FILL THIS OUT LATER)
            else:
                return {'outcome': 'NoPIIDetected'}
        return {'outcome': 'UNKNOWN'}

    def prepare(self):

    def initialize(self):

    def execute(self):

        self.output = self._populate_data(results)
        self.confidence = self._calculate_confidence()

        bt.logging.debug(
            f"PII engine executed (Confidence: {self.confidence} - Output: {self.output})"
        )