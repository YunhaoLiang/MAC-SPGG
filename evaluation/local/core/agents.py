"""
Local model agents for SPGG GSM8K evaluation.

All models run locally without any API calls.
Supported models:
- Llama: meta-llama/Llama-3.1-8B-Instruct
- SmolLM2: HuggingFaceTB/SmolLM2-1.7B-Instruct
- Qwen: Qwen/Qwen2.5-7B-Instruct
"""

import os
import re
import time
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import parse_generation_params, extract_numerical_answer


# Local model configurations
MODEL_CONFIGS = {
    "LLAMA": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "torch_dtype": torch.float16,
        "max_tokens_default": 1024,
    },
    "SMOLLM2": {
        "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "torch_dtype": torch.float16,
        "max_tokens_default": 512,
    },
    "QWEN": {
        "name": "Qwen/Qwen3-8B",
        "torch_dtype": torch.float16,
        "max_tokens_default": 1024,
    }
}

# Agent to model mapping
AGENT_MODEL_MAP = {
    "Agent_Llama": "LLAMA",
    "Agent_SMOLLM2": "SMOLLM2",
    "Agent_Qwen": "QWEN"
}


class LocalModelManager:
    """
    Manager for local language models.
    
    Handles loading and inference for all local HuggingFace models.
    Supports dynamic parameter generation from trained policy networks.
    
    Attributes:
        models: Dictionary of loaded models and tokenizers.
        dynamic_generators: Dictionary of policy networks for parameter generation.
    """
    
    def __init__(self):
        """Initialize the local model manager."""
        self.models: Dict[str, Dict] = {}
        self.dynamic_generators: Dict[str, Dict] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LocalModelManager initialized, device: {self.device}")
    
    def set_dynamic_generators(self, generators: Dict[str, Dict]) -> None:
        """
        Set the dynamic parameter generators for each agent.
        
        Args:
            generators: Dictionary mapping agent IDs to generator configurations.
        """
        self.dynamic_generators = generators
        dynamic_count = sum(1 for info in generators.values() if info.get('is_dynamic', False))
        print(f"Dynamic parameter generators configured: {dynamic_count}/{len(generators)} agents enabled")
    
    def load_model(self, model_key: str) -> bool:
        """
        Load a local model by key.
        
        Args:
            model_key: Key identifying the model (LLAMA, SMOLLM2, QWEN).
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if model_key in self.models:
            return True
        
        if model_key not in MODEL_CONFIGS:
            print(f"Error: Unknown model key: {model_key}")
            return False
        
        config = MODEL_CONFIGS[model_key]
        model_name = config["name"]
        torch_dtype = config["torch_dtype"]
        
        try:
            print(f"Loading model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config
            }
            
            print(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def load_all_models(self) -> bool:
        """
        Load all required models.
        
        Returns:
            True if all models loaded successfully, False otherwise.
        """
        success = True
        for model_key in MODEL_CONFIGS.keys():
            if not self.load_model(model_key):
                success = False
        return success
    
    def generate_dynamic_params(
        self,
        agent_id: str,
        math_problem: str,
        position: int = 0,
        previous_solutions: Optional[List[str]] = None,
        problem_index: int = 0
    ) -> Dict[str, Any]:
        """
        Generate LLM parameters using the trained policy network.
        
        Args:
            agent_id: Identifier of the agent.
            math_problem: The mathematical problem text.
            position: Position in the sequential game.
            previous_solutions: Previous agent solutions.
            problem_index: Index of current problem (for logging).
        
        Returns:
            Dictionary of generation parameters.
        """
        if agent_id not in self.dynamic_generators:
            return self._get_default_params(agent_id)
        
        generator_info = self.dynamic_generators[agent_id]
        
        if not generator_info.get('is_dynamic', False):
            return {k: v for k, v in generator_info.items() if k != 'is_dynamic'}
        
        try:
            policy_network = generator_info['policy_network']
            state_encoder = generator_info['state_encoder']
            config = generator_info['config']
            
            state = state_encoder.encode_state(
                math_problem, agent_id, position, previous_solutions
            ).unsqueeze(0)
            
            with torch.no_grad():
                raw_params = policy_network(state).squeeze(0)
                dynamic_params = parse_generation_params(raw_params, config, agent_id)
            
            if problem_index < 5:
                print(f"{agent_id} dynamic params: temp={dynamic_params['temperature']:.3f}, "
                      f"top_p={dynamic_params['top_p']:.3f}, max_tokens={dynamic_params['max_tokens']}")
            
            return dynamic_params
            
        except Exception as e:
            print(f"Error: {agent_id} dynamic parameter generation failed: {e}")
            return self._get_default_params(agent_id)
    
    def _get_default_params(self, agent_id: str) -> Dict[str, Any]:
        """Get default parameters for an agent."""
        model_key = AGENT_MODEL_MAP.get(agent_id, "QWEN")
        config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["QWEN"])
        
        return {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'max_tokens': config["max_tokens_default"],
            'repetition_penalty': 1.1,
            'is_dynamic_params': False
        }
    
    def generate_response(
        self,
        agent_id: str,
        messages: List[Dict],
        math_problem: str = "",
        position: int = 0,
        previous_solutions: Optional[List[str]] = None,
        problem_index: int = 0
    ) -> str:
        """
        Generate response using local model.
        
        Args:
            agent_id: Agent identifier.
            messages: Chat messages.
            math_problem: Problem text for state encoding.
            position: Position in sequential game.
            previous_solutions: Previous agent solutions.
            problem_index: Problem index for logging.
        
        Returns:
            Generated response text, or empty string on failure.
        """
        model_key = AGENT_MODEL_MAP.get(agent_id)
        if not model_key:
            print(f"Error: Unknown agent: {agent_id}")
            return ""
        
        if model_key not in self.models:
            if not self.load_model(model_key):
                return ""
        
        try:
            model_data = self.models[model_key]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            dynamic_params = self.generate_dynamic_params(
                agent_id, math_problem, position, previous_solutions, problem_index
            )
            
            # Apply chat template
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = f"User: {messages[-1]['content']}\nAssistant:"
            
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=dynamic_params.get('max_tokens', 512),
                    temperature=max(dynamic_params.get('temperature', 0.7), 0.01),
                    top_p=dynamic_params.get('top_p', 0.9),
                    top_k=dynamic_params.get('top_k', 50),
                    repetition_penalty=dynamic_params.get('repetition_penalty', 1.1),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            return response.strip()
            
        except torch.cuda.OutOfMemoryError:
            print(f"Warning: GPU OOM for {agent_id}, trying with reduced tokens")
            try:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                return response.strip()
            except Exception as e:
                print(f"Error: Reduced generation also failed: {e}")
                return ""
        except Exception as e:
            print(f"Error: Local model generation failed for {agent_id}: {e}")
            return ""


class MathProblemPool:
    """
    Pool for collecting and managing solutions in the public goods game.
    
    Attributes:
        individual_solutions: Dictionary mapping agent IDs to their solutions.
        contribution_history: List of contribution events with timestamps.
        final_solution: The final integrated solution.
        correct_answer: The ground truth answer.
    """
    
    def __init__(self):
        """Initialize an empty problem pool."""
        self.individual_solutions: Dict[str, str] = {}
        self.contribution_history: List[Dict] = []
        self.final_solution: str = ""
        self.correct_answer: str = ""
    
    def set_correct_answer(self, answer: str) -> None:
        """Set the ground truth answer."""
        self.correct_answer = answer
    
    def add_individual_solution(
        self,
        agent_id: str,
        solution: str,
        problem_index: int = 0
    ) -> None:
        """Add an individual agent's solution to the pool."""
        self.individual_solutions[agent_id] = solution
        self.contribution_history.append({
            "agent_id": agent_id,
            "timestamp": time.time(),
            "solution_type": "individual",
            "content": solution
        })
    
    def get_all_individual_solutions(self) -> str:
        """Get a formatted summary of all individual solutions."""
        if not self.individual_solutions:
            return "No individual solutions available yet."
        
        summary = "=== Individual Solutions from All Agents ===\n\n"
        for agent_id, solution in self.individual_solutions.items():
            summary += f"--- Solution from {agent_id} ---\n"
            summary += solution
            summary += "\n\n"
        
        return summary
    
    def extract_answer(self, text: str, show_debug: bool = False) -> str:
        """Extract numerical answer from solution text."""
        return extract_numerical_answer(text, show_debug)


class MathAgent:
    """
    Agent for mathematical reasoning using local models.
    
    Attributes:
        agent_id: Unique identifier for the agent.
        model_manager: LocalModelManager instance for model calls.
    """
    
    def __init__(self, agent_id: str, model_manager: LocalModelManager):
        """
        Initialize a math agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            model_manager: LocalModelManager instance.
        """
        self.agent_id = agent_id
        self.model_manager = model_manager
    
    def generate_solution(
        self,
        problem: str,
        previous_solution: Optional[Dict[str, str]] = None,
        position: int = 0,
        problem_index: int = 0,
        is_last_agent: bool = False
    ) -> str:
        """
        Generate a solution for the given math problem (partial observation).
        
        Args:
            problem: The mathematical problem text.
            previous_solution: Solution from the previous agent.
            position: Position in the sequential game.
            problem_index: Index of current problem.
            is_last_agent: Whether this is the final agent.
        
        Returns:
            Generated solution text.
        """
        prompt = self._build_prompt(problem, previous_solution, is_last_agent)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        previous_solution_texts = [previous_solution["solution"]] if previous_solution else []
        
        return self.model_manager.generate_response(
            self.agent_id,
            messages,
            math_problem=problem,
            position=position,
            previous_solutions=previous_solution_texts,
            problem_index=problem_index
        )
    
    def _build_prompt(
        self,
        problem: str,
        previous_solution: Optional[Dict[str, str]] = None,
        is_last_agent: bool = False
    ) -> str:
        """Build the prompt for the agent (partial observation)."""
        if is_last_agent:
            if previous_solution:
                return f"""Question: {problem}

Previous agent's solution:
--- Solution from {previous_solution['agent_id']} ---
{previous_solution['solution']}
Extracted answer: {previous_solution['extracted_answer']}

You are the FINAL agent in this Partial Observation + No Integrator setup. Your answer will be the FINAL answer.
You can see the previous agent's solution and reasoning.

Instructions:
1. Analyze the previous solution carefully
2. Consider the approach and identify any potential errors
3. Provide your own independent solution
4. Give your final answer that will be used as the group's answer

Give your final answer in the format "The answer is [number]"."""
            else:
                return f"""Question: {problem}

You are the FINAL agent. Solve this step by step and provide your final answer.
Give your final answer in the format "The answer is [number]"."""
        else:
            if previous_solution:
                return f"""Question: {problem}

Previous agent's solution:
--- Solution from {previous_solution['agent_id']} ---
{previous_solution['solution']}
Extracted answer: {previous_solution['extracted_answer']}

You can see the previous agent's solution above. Please provide your own independent analysis and solution.
Consider the previous approach but think critically and provide your own reasoning.

Give your final answer in the format "The answer is [number]"."""
            else:
                return f"""Question: {problem}

Please solve this step by step and provide your final answer in the format "The answer is [number]"."""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return f"""You are {self.agent_id}, participating in a math problem Public Goods Game (Partial Observation + No Integrator Mode).

In this mode:
- You can see ONLY the previous agent's complete solution (Partial Observation)
- The last agent's answer becomes the final group answer (No Integrator)
- All participants share the same reward based on the final answer's correctness

Solve the problem independently and accurately, providing precise reasoning and a clear final answer. Always conclude your response with "The answer is [number]"."""
