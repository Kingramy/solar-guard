# agent/llm.py
import json
from pathlib import Path

class MockLLM:
    def __init__(self, fixture_path: str = "agent/fixtures/mock_llm_responses.json"):
        path = Path(fixture_path)
        if not path.exists():
            fallback = Path(__file__).resolve().parent / "fixtures" / "mock_llm_responses.json"
            if fallback.exists():
                path = fallback
        self.fixture_path = path
        if self.fixture_path.exists():
            with open(self.fixture_path, "r") as f:
                self.responses = json.load(f)
        else:
            self.responses = {}
        self.call_count = 0

    def chat(self, context: str, scene_id: str, detection_id: str) -> dict:
        key = f"{scene_id}_{detection_id}"
        sequence = self.responses.get(key, [])
        
        if self.call_count < len(sequence):
            resp = sequence[self.call_count]
            self.call_count += 1
            return resp
            
        # Safe fallback if fixture runs out
        return {
            "text": "Trace complete. Deferring to deterministic guardrail.", 
            "tool_calls": [],
            "proposed_decision": "MONITOR_ONLY"
        }
        
    def reset(self):
        self.call_count = 0