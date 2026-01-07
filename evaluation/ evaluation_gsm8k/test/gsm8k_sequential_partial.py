#!/usr/bin/env python3
"""
GSM8K Evaluation with SPGG Framework - Partial Observation Mode.

This script evaluates the Sequential Public Goods Game (SPGG) framework
on the GSM8K mathematical reasoning benchmark. It implements the partial
observation mode where each agent can only see the immediately preceding
agent's solution, combined with the no-integrator mode where the final
agent's answer serves as the group answer.

Agent sequence: Llama -> SmolLM2 -> Qwen

"""

import os
import sys
import re
import time
import json
import logging

# Add SPGG root directory to path for imports
# test/ -> evaluation_gsm8k/ -> evaluation/ -> SPGG/
SPGG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, SPGG_ROOT)

from huggingface_hub import login

from core import (
    PolicyNetwork,
    ValueNetwork,
    MathStateEncoder,
    SPGGConfig,
    parse_generation_params,
    load_checkpoint_params,
    load_gsm8k_samples,
    safe_for_json,
    save_results,
    APIModelManager,
    MathProblemPool,
    MathAgent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace authentication
try:
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if hf_token:
        login(hf_token)
except Exception:
    pass

# Result directory
RESULT_DIR = "/content/drive/MyDrive/PGG/GSM8K"
os.makedirs(RESULT_DIR, exist_ok=True)


# Mock module for checkpoint compatibility
class MockSPGGModule:
    """Mock module for loading checkpoints without full SPGG dependencies."""
    SPGGConfig = SPGGConfig

sys.modules['spgg_multi_agent_trainer'] = MockSPGGModule


def run_gsm8k_partial_observation(
    problem: dict,
    api_manager: APIModelManager,
    verbose: bool = True,
    problem_index: int = 0
) -> dict:
    """
    Run SPGG evaluation on a single GSM8K problem.
    
    Executes the sequential public goods game with partial observation,
    where each agent sees only the previous agent's solution. The final
    agent (Qwen) provides the group's answer.
    
    Args:
        problem: Dictionary containing 'question' and 'answer' keys.
        api_manager: APIModelManager instance for LLM calls.
        verbose: Whether to print detailed progress information.
        problem_index: Index of the current problem.
    
    Returns:
        Dictionary containing:
            - question: The original problem text.
            - correct_answer: Ground truth answer.
            - extracted_answer: Extracted answer from final solution.
            - is_correct: Boolean indicating correctness.
            - individual_solutions: Dict mapping agent IDs to solutions.
            - final_solution: The final agent's solution.
            - failed_agents: List of agents that failed to generate.
            - mode: Evaluation mode identifier.
            - agent_models: Model specifications for each agent.
            - summary: Summary statistics.
    """
    question = problem['question']
    correct_answer = problem['answer']
    
    # Show detailed output for first few problems
    show_detailed = problem_index < 3
    
    if show_detailed:
        print(f"\n{'='*60}")
        print(f"Problem {problem_index + 1} - Detailed View")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        print(f"{'='*60}")
    else:
        print(f"Problem {problem_index + 1}: {question[:50]}...")
        print(f"Correct Answer: {correct_answer}")
    
    # Initialize problem pool
    math_pool = MathProblemPool()
    math_pool.set_correct_answer(correct_answer)
    
    # Initialize agents: Llama -> SmolLM2 -> Qwen
    agents = [
        MathAgent("Agent_Llama", "together", api_manager),
        MathAgent("Agent_SMOLLM2", "local", api_manager),
        MathAgent("Agent_Qwen", "qwen", api_manager)
    ]
    
    # Load local model
    if not api_manager.load_local_model("SMOLLM2"):
        print("Error: SmolLM2 model loading failed")
        return {
            "question": question,
            "error": "SmolLM2 model loading failed"
        }
    
    print("Agent configuration: Llama -> SmolLM2 -> Qwen")
    
    all_solutions = []
    failed_agents = []
    previous_solution = None
    
    for i, agent in enumerate(agents):
        is_last_agent = (i == len(agents) - 1)
        
        if show_detailed:
            print(f"\n{agent.agent_id} generating solution...")
            if previous_solution:
                print(f"   Visible solution: {previous_solution['agent_id']} (partial observation)")
            else:
                print(f"   Visible solution: None (first agent)")
            if is_last_agent:
                print(f"   This is the final agent - will provide group answer")
        else:
            print(f"{agent.agent_id} generating solution...")
        
        solution = agent.generate_solution(
            question,
            previous_solution,
            position=i+1,
            problem_index=problem_index,
            is_last_agent=is_last_agent
        )
        
        if solution:
            if show_detailed:
                print(f"{agent.agent_id} completed")
                print(f"   Response: {solution}")
            else:
                print(f"{agent.agent_id} completed, response length: {len(solution)}")
            
            math_pool.add_individual_solution(agent.agent_id, solution, problem_index)
            
            extracted_answer = math_pool.extract_answer(solution, show_debug=False)
            
            current_solution = {
                "agent_id": agent.agent_id,
                "solution": solution,
                "extracted_answer": extracted_answer
            }
            all_solutions.append(current_solution)
            
            # Partial observation: only pass current solution to next agent
            previous_solution = current_solution
            
            if show_detailed:
                print(f"   Extracted answer: {extracted_answer}")
                print(f"   Solution length: {len(solution)}")
            else:
                print(f"{agent.agent_id} extracted answer: {extracted_answer}")
        else:
            print(f"{agent.agent_id} failed to generate solution")
            failed_agents.append(agent.agent_id)
    
    # No integrator mode: final answer from last successful agent
    final_solution = all_solutions[-1]["solution"] if all_solutions else ""
    final_answer_extracted = all_solutions[-1]["extracted_answer"] if all_solutions else "No answer found"
    
    math_pool.final_solution = final_solution
    
    # Evaluate correctness
    is_correct = False
    try:
        extracted_num = float(re.sub(r'[^\d.]', '', final_answer_extracted))
        correct_num = float(re.sub(r'[^\d.]', '', correct_answer))
        is_correct = abs(extracted_num - correct_num) < 0.01
    except Exception:
        is_correct = final_answer_extracted.strip() == correct_answer.strip()
    
    if show_detailed:
        print(f"\n{'='*60}")
        print(f"Final Answer: {final_answer_extracted}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
        print(f"{'='*60}")
    elif verbose:
        print(f"Problem {problem_index + 1}: {final_answer_extracted}")
    
    return {
        "question": question,
        "correct_answer": correct_answer,
        "extracted_answer": final_answer_extracted,
        "is_correct": is_correct,
        "individual_solutions": dict(math_pool.individual_solutions),
        "final_solution": final_solution,
        "failed_agents": failed_agents,
        "mode": "partial_observation_no_integrator",
        "agent_models": {
            "Agent_Llama": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "Agent_SMOLLM2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Agent_Qwen": "qwen3-8b"
        },
        "summary": {
            "contributions_count": len(math_pool.individual_solutions),
            "final_solution_length": len(final_solution),
            "success": is_correct,
            "mode": "partial_observation_no_integrator"
        }
    }


def main():
    """
    Main entry point for GSM8K SPGG evaluation.
    
    Loads the trained checkpoint, initializes the API manager with
    dynamic parameter generators, and runs evaluation on all GSM8K problems.
    """
    # Checkpoint path (relative to SPGG root)
    checkpoint_path = os.path.join(SPGG_ROOT, "checkpoints", "checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    learned_params = load_checkpoint_params(checkpoint_path)
    
    # Initialize API manager
    api_manager = APIModelManager()
    api_manager.set_dynamic_generators(learned_params)
    
    # Load GSM8K data
    gsm8k_data_path = "/content/drive/MyDrive/PGG/GSM8K/Dataset/gsm8k_samples.json"
    
    if not os.path.exists(gsm8k_data_path):
        print(f"Error: GSM8K data file not found at {gsm8k_data_path}")
        print("Please ensure the data file is uploaded to Google Drive")
        return
    
    gsm8k_samples = load_gsm8k_samples(gsm8k_data_path)
    if not gsm8k_samples:
        print("Error: Failed to load GSM8K data")
        return
    
    # Run evaluation
    results = []
    total_correct = 0
    total_wrong = 0
    
    print(f"Starting evaluation on {len(gsm8k_samples)} GSM8K problems")
    print(f"Mode: Partial Observation + No Integrator")
    print(f"Agent Sequence: Llama -> SmolLM2 -> Qwen")
    
    for idx, problem in enumerate(gsm8k_samples):
        print(f"\nProcessing problem {idx + 1}...")
        print(f"Question: {problem['question'][:100]}...")
        
        try:
            result = run_gsm8k_partial_observation(
                problem, api_manager, verbose=True, problem_index=idx
            )
            results.append(result)
            
            if result.get('is_correct', False):
                total_correct += 1
            else:
                total_wrong += 1
                
        except Exception as e:
            print(f"Error processing problem {idx + 1}: {e}")
            results.append({
                "question": problem.get('question', ''),
                "error": str(e)
            })
            total_wrong += 1
        
        print(f"Problem {idx + 1} completed")
        time.sleep(0.5)
    
    # Compute statistics
    processed_problems = len(results)
    accuracy = total_correct / processed_problems if processed_problems else 0
    
    # Clean parameters for JSON serialization
    clean_learned_params = {}
    for agent_id, params in learned_params.items():
        if isinstance(params, dict):
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    clean_params[key] = value
                elif key in ['temperature', 'top_p', 'max_tokens', 'repetition_penalty', 'is_dynamic']:
                    clean_params[key] = value
            clean_learned_params[agent_id] = clean_params
        else:
            clean_learned_params[agent_id] = str(params)
    
    # Compile final results
    final_results = {
        "experiment_info": {
            "total_problems": len(gsm8k_samples),
            "processed_problems": processed_problems,
            "correct_answers": total_correct,
            "wrong_answers": total_wrong,
            "accuracy": accuracy,
            "mode": "partial_observation_no_integrator",
            "observation_mode": "Each agent can see ONLY the previous agent's solution",
            "integrator_mode": "No Integrator - Final agent's answer is the group answer",
            "agent_sequence": "Llama -> SmolLM2 -> Qwen",
            "checkpoint_used": checkpoint_path if os.path.exists(checkpoint_path) else "default_params",
            "learned_parameters": clean_learned_params,
            "environment": "Partial Observation + No Integrator - Mixed API/Local",
            "model_details": {
                "Agent_Llama": "meta-llama/Llama-3.1-8B-Instruct-Turbo (Together API)",
                "Agent_SMOLLM2": "HuggingFaceTB/SmolLM2-1.7B-Instruct (Local)",
                "Agent_Qwen": "qwen3-8b (DashScope API)"
            }
        },
        "results": results
    }
    
    # Print summary
    print(f"\n===== Evaluation Results =====")
    print(f"Agent Sequence: Llama -> SmolLM2 -> Qwen")
    print(f"Observation Mode: Partial (each agent sees only previous agent's solution)")
    print(f"Total Problems: {processed_problems}")
    print(f"Correct: {total_correct}")
    print(f"Wrong: {total_wrong}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results
    save_results([final_results], RESULT_DIR, "partial_obs_no_integrator")
    print(f"Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
