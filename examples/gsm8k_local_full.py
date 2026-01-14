#!/usr/bin/env python3
"""
GSM8K Evaluation with SPGG Framework - Local Models Only (Full Observation).

All models run locally without any API calls.
Agent sequence: Llama -> SmolLM2 -> Qwen (all local)

Full observation mode: each agent can see ALL previous agents' solutions.
"""

import os
import sys
import re
import time
import json
import logging

# Add parent directory to path for imports
SPGG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPGG_ROOT)

from huggingface_hub import login

# Import from local core modules
from spgg.evaluation.local.core import (
    PolicyNetwork,
    ValueNetwork,
    MathStateEncoder,
    SPGGConfig,
    parse_generation_params,
    load_checkpoint_params,
    load_gsm8k_samples,
    safe_for_json,
    save_results,
    LocalModelManager,
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
RESULT_DIR = "/content/drive/MyDrive/PGG/GSM8K/local"
os.makedirs(RESULT_DIR, exist_ok=True)


# Mock module for checkpoint compatibility
class MockSPGGModule:
    """Mock module for loading checkpoints without full SPGG dependencies."""
    SPGGConfig = SPGGConfig

sys.modules['spgg_multi_agent_trainer'] = MockSPGGModule


class FullObservationMathAgent(MathAgent):
    """
    Math agent for full observation mode using local models.
    
    In full observation mode, each agent can see ALL previous agents'
    solutions, not just the immediately preceding one.
    """
    
    def generate_solution(
        self,
        problem: str,
        all_previous_solutions: list = None,
        position: int = 0,
        problem_index: int = 0,
        is_last_agent: bool = False
    ) -> str:
        """
        Generate a solution with full observation of all previous solutions.
        
        Args:
            problem: The mathematical problem text.
            all_previous_solutions: List of all previous agents' solutions.
            position: Position in the sequential game.
            problem_index: Index of current problem.
            is_last_agent: Whether this is the final agent.
        
        Returns:
            Generated solution text.
        """
        prompt = self._build_full_obs_prompt(problem, all_previous_solutions, is_last_agent)
        messages = [
            {"role": "system", "content": self._get_full_obs_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        previous_solution_texts = [sol["solution"] for sol in all_previous_solutions] if all_previous_solutions else []
        
        return self.model_manager.generate_response(
            self.agent_id,
            messages,
            math_problem=problem,
            position=position,
            previous_solutions=previous_solution_texts,
            problem_index=problem_index
        )
    
    def _build_full_obs_prompt(
        self,
        problem: str,
        all_previous_solutions: list = None,
        is_last_agent: bool = False
    ) -> str:
        """Build the prompt for full observation mode."""
        if is_last_agent:
            if all_previous_solutions:
                previous_content = ""
                for sol in all_previous_solutions:
                    previous_content += f"--- Solution from {sol['agent_id']} ---\n"
                    previous_content += f"{sol['solution']}\n"
                    previous_content += f"Extracted answer: {sol['extracted_answer']}\n\n"
                
                return f"""Question: {problem}

Previous agents' solutions:
{previous_content}

You are the FINAL agent in this Full Observation + No Integrator setup. Your answer will be the FINAL answer.
You can see ALL previous agents' solutions and their reasoning.

Instructions:
1. Analyze ALL previous solutions carefully
2. Compare their approaches and identify any errors
3. Provide your own independent solution
4. Give your final answer that will be used as the group's answer

Give your final answer in the format "The answer is [number]"."""
            else:
                return f"""Question: {problem}

You are the FINAL agent. Solve this step by step and provide your final answer.
Give your final answer in the format "The answer is [number]"."""
        else:
            if all_previous_solutions:
                previous_content = ""
                for sol in all_previous_solutions:
                    previous_content += f"--- Solution from {sol['agent_id']} ---\n"
                    previous_content += f"{sol['solution']}\n"
                    previous_content += f"Extracted answer: {sol['extracted_answer']}\n\n"
                
                return f"""Question: {problem}

Previous agents' solutions:
{previous_content}

You can see ALL previous agents' solutions above. Please provide your own independent analysis and solution.
Consider the previous approaches but think critically and provide your own reasoning.

Give your final answer in the format "The answer is [number]"."""
            else:
                return f"""Question: {problem}

Please solve this step by step and provide your final answer in the format "The answer is [number]"."""
    
    def _get_full_obs_system_prompt(self) -> str:
        """Get the system prompt for full observation mode."""
        return f"""You are {self.agent_id}, participating in a math problem Public Goods Game (Full Observation + No Integrator Mode).

In this mode:
- You can see ALL previous agents' complete solutions (Full Observation)
- The last agent's answer becomes the final group answer (No Integrator)
- All participants share the same reward based on the final answer's correctness

Solve the problem independently and accurately, providing precise reasoning and a clear final answer. Always conclude your response with "The answer is [number]"."""


def run_gsm8k_full_observation(
    problem: dict,
    model_manager: LocalModelManager,
    verbose: bool = True,
    problem_index: int = 0
) -> dict:
    """
    Run SPGG evaluation on a single GSM8K problem with full observation.
    
    Args:
        problem: Dictionary containing 'question' and 'answer' keys.
        model_manager: LocalModelManager instance for model calls.
        verbose: Whether to print detailed progress information.
        problem_index: Index of the current problem.
    
    Returns:
        Dictionary containing evaluation results.
    """
    question = problem['question']
    correct_answer = problem['answer']
    
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
    
    math_pool = MathProblemPool()
    math_pool.set_correct_answer(correct_answer)
    
    # Initialize agents: Llama -> SmolLM2 -> Qwen (all local)
    agents = [
        FullObservationMathAgent("Agent_Llama", model_manager),
        FullObservationMathAgent("Agent_SMOLLM2", model_manager),
        FullObservationMathAgent("Agent_Qwen", model_manager)
    ]
    
    print("Agent configuration: Llama -> SmolLM2 -> Qwen (all local)")
    
    all_solutions = []
    failed_agents = []
    
    for i, agent in enumerate(agents):
        is_last_agent = (i == len(agents) - 1)
        
        if show_detailed:
            print(f"\n{agent.agent_id} generating solution...")
            print(f"   Visible solutions: {len(all_solutions)} (all previous agents)")
            if is_last_agent:
                print(f"   This is the final agent - will provide group answer")
        else:
            print(f"{agent.agent_id} generating solution...")
        
        solution = agent.generate_solution(
            question,
            all_solutions,
            position=i+1,
            problem_index=problem_index,
            is_last_agent=is_last_agent
        )
        
        if solution:
            if show_detailed:
                print(f"{agent.agent_id} completed")
                print(f"   Response: {solution[:200]}...")
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
            
            if show_detailed:
                print(f"   Extracted answer: {extracted_answer}")
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
        "mode": "local_full_observation_no_integrator",
        "agent_models": {
            "Agent_Llama": "meta-llama/Llama-3.1-8B-Instruct (local)",
            "Agent_SMOLLM2": "HuggingFaceTB/SmolLM2-1.7B-Instruct (local)",
            "Agent_Qwen": "Qwen/Qwen3-8B (local)"
        },
        "summary": {
            "contributions_count": len(math_pool.individual_solutions),
            "final_solution_length": len(final_solution),
            "success": is_correct,
            "mode": "local_full_observation_no_integrator"
        }
    }


def main():
    """Main entry point for local GSM8K SPGG evaluation with full observation."""
    # Checkpoint path (relative to SPGG root)
    # This checkpoint was trained under PO protocol.
    # Using it here for Full Observation is an ablation/generalization test to evaluate the checkpoint's transferability across different information settings.
    checkpoint_path = os.path.join(SPGG_ROOT, "spgg", "checkpoints", "checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    learned_params = load_checkpoint_params(checkpoint_path)
    
    # Initialize local model manager
    model_manager = LocalModelManager()
    model_manager.set_dynamic_generators(learned_params)
    
    # Pre-load all models
    print("Pre-loading all local models...")
    if not model_manager.load_all_models():
        print("Warning: Some models failed to load")
    
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
    print(f"Mode: Local Models + Full Observation + No Integrator")
    print(f"Agent Sequence: Llama -> SmolLM2 -> Qwen (all local)")
    
    for idx, problem in enumerate(gsm8k_samples):
        print(f"\nProcessing problem {idx + 1}...")
        print(f"Question: {problem['question'][:100]}...")
        
        try:
            result = run_gsm8k_full_observation(
                problem, model_manager, verbose=True, problem_index=idx
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
            "mode": "local_full_observation_no_integrator",
            "observation_mode": "Each agent can see ALL previous agents' solutions",
            "integrator_mode": "No Integrator - Final agent's answer is the group answer",
            "agent_sequence": "Llama -> SmolLM2 -> Qwen (all local)",
            "checkpoint_used": checkpoint_path,
            "learned_parameters": clean_learned_params,
            "environment": "All Local Models - No API",
            "model_details": {
                "Agent_Llama": "meta-llama/Llama-3.1-8B-Instruct (local)",
                "Agent_SMOLLM2": "HuggingFaceTB/SmolLM2-1.7B-Instruct (local)",
                "Agent_Qwen": "Qwen/Qwen3-8B (local)"
            }
        },
        "results": results
    }
    
    # Print summary
    print(f"\n===== Evaluation Results =====")
    print(f"Agent Sequence: Llama -> SmolLM2 -> Qwen (all local)")
    print(f"Observation Mode: Full (each agent sees all previous agents' solutions)")
    print(f"Total Problems: {processed_problems}")
    print(f"Correct: {total_correct}")
    print(f"Wrong: {total_wrong}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results
    save_results([final_results], RESULT_DIR, "local_full_obs")
    print(f"Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
