# agent/agent.py
from typing import Dict, Any, List
try:
    from .tools import TOOL_REGISTRY
    from .llm import MockLLM
    from .policy import evaluate_panel_safety_and_roi
except (ImportError, ModuleNotFoundError):
    from tools import TOOL_REGISTRY
    from llm import MockLLM
    from policy import evaluate_panel_safety_and_roi

def run_agent_loop(detection: Dict[str, Any], scene_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes a bounded ReAct loop. Independently checks the deterministic policy
    and overrides the LLM's proposal if it hallucinates or acts unsafely.
    """
    llm = MockLLM()
    trace = []
    
    context = f"Analyze panel {detection['id']}. Class: {detection['ml_class']}. Temp: {detection['max_temp_c']}C."
    
    # 1. Bounded ReAct Loop (Max 8 iterations)
    for i in range(8):
        response = llm.chat(context, scene_id, detection["id"])
        
        # Log the LLM's thought process
        trace.append({"event": "rationale", "content": response["text"]})
        
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            llm_proposed_decision = response.get("proposed_decision", "MONITOR_ONLY")
            break
            
        # Execute requested tools
        for call in tool_calls:
            tool_name = call["name"]
            args = call["arguments"]
            
            trace.append({"event": "action", "content": f"Calling {tool_name} with {args}"})
            
            if tool_name in TOOL_REGISTRY:
                if tool_name == "check_safety":
                    obs = TOOL_REGISTRY[tool_name](args["max_temp_c"], args["delta_t_c"], args["category"], config)
                else:
                    obs = TOOL_REGISTRY[tool_name](**args)
                trace.append({"event": "observation", "content": str(obs)})
            else:
                trace.append({"event": "observation", "content": f"Error: Tool {tool_name} not found."})

    # 2. Independent Deterministic Guardrail Calculation
    policy_decision = evaluate_panel_safety_and_roi(detection, config)
    
    # 3. Guardrail Intercept
    final_decision = policy_decision
    if llm_proposed_decision != policy_decision:
        trace.append({
            "event": "guardrail_override", 
            "content": f"LLM proposed {llm_proposed_decision}, but deterministic policy mandates {policy_decision}. Override applied."
        })
        
    trace.append({"event": "final", "content": final_decision})
    
    return trace

if __name__ == "__main__":
    try:
        from .loss_model import load_config
    except (ImportError, ModuleNotFoundError):
        from loss_model import load_config
    config = load_config()
    
    # Let's test Demo Step 5: The D-0002 Hotspot disguised as "No-Anomaly"
    test_detection = {
        "id": "D-0002",
        "ml_class": "No-Anomaly",
        "agent_category": "none",
        "max_temp_c": 95.0,
        "delta_t_c": 54.8,
        "flag_reason": "temperature"
    }
    
    print("\n--- Running Agentic Trace for Panel D-0002 (95.0C Healthy Injection) ---")
    trace = run_agent_loop(test_detection, "scene_critical_01", config)
    
    for step in trace:
        print(f"[{step['event'].upper()}] {step['content']}")