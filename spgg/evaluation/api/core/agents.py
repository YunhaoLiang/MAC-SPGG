"""
Agent implementations for SPGG GSM8K evaluation.

Defines the mathematical reasoning agents, API managers, and problem pool
for the sequential public goods game on GSM8K benchmark.
"""

import os
import re
import time
import json
import torch
import requests
import certifi
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import parse_generation_params, extract_numerical_answer


# API configuration
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

# Model configuration
SMOLLM2_CONFIG = {
    "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "cache_dir": None,
    "torch_dtype": torch.float16
}


class APIModelManager:
    """
    Manager for API-based and local language models.
    
    Handles communication with Together AI, Alibaba DashScope, and local
    HuggingFace models. Supports dynamic parameter generation from trained
    policy networks.
    
    Attributes:
        local_models: Dictionary of loaded local models.
        dynamic_generators: Dictionary of policy networks for parameter generation.
    """
    
    def __init__(self):
        """Initialize the API model manager."""
        self.local_models: Dict[str, Dict] = {}
        self.dynamic_generators: Dict[str, Dict] = {}
    
    def set_dynamic_generators(self, generators: Dict[str, Dict]) -> None:
        """
        Set the dynamic parameter generators for each agent.
        
        Args:
            generators: Dictionary mapping agent IDs to generator configurations.
        """
        self.dynamic_generators = generators
        dynamic_count = sum(1 for info in generators.values() if info.get('is_dynamic', False))
        print(f"Dynamic parameter generators configured: {dynamic_count}/{len(generators)} agents enabled")
    
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
            previous_solutions: Previous agent solutions (partial observation).
            problem_index: Index of current problem (for logging).
        
        Returns:
            Dictionary of generation parameters.
        """
        if agent_id not in self.dynamic_generators:
            print(f"Error: {agent_id} has no dynamic generator")
            return {}
        
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
            
            return dynamic_params
            
        except Exception as e:
            print(f"Error: {agent_id} dynamic parameter generation failed: {e}")
            return self._get_default_params(agent_id)
    
    def _get_default_params(self, agent_id: str) -> Dict[str, Any]:
        """Get default parameters for an agent when dynamic generation fails."""
        defaults = {
            "Agent_Llama": {
                'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 1012,
                'repetition_penalty': 1.1, 'is_dynamic_params': False
            },
            "Agent_SMOLLM2": {
                'temperature': 0.8, 'top_p': 0.85, 'max_tokens': 256,
                'repetition_penalty': 1.15, 'is_dynamic_params': False
            },
            "Agent_Qwen": {
                'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 1012,
                'repetition_penalty': 1.1, 'is_dynamic_params': False
            }
        }
        return defaults.get(agent_id, defaults["Agent_Qwen"])
    
    def call_together_api(
        self,
        model_name: str,
        messages: List[Dict],
        agent_id: str = "",
        math_problem: str = "",
        position: int = 0,
        previous_solutions: Optional[List[str]] = None,
        problem_index: int = 0,
        **generation_params
    ) -> str:
        """
        Call Together AI API for text generation.
        
        Args:
            model_name: Name of the model to use.
            messages: Chat messages in OpenAI format.
            agent_id: Agent identifier for dynamic parameters.
            math_problem: Problem text for state encoding.
            position: Position in sequential game.
            previous_solutions: Previous agent solutions.
            problem_index: Problem index for logging.
            **generation_params: Additional generation parameters.
        
        Returns:
            Generated response text, or empty string on failure.
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = "https://api.together.xyz/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                agent_id_for_params = agent_id if agent_id else 'Agent_Llama'
                dynamic_params = self.generate_dynamic_params(
                    agent_id_for_params, math_problem, position, previous_solutions, problem_index
                )
                
                payload = {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "messages": messages,
                    "temperature": dynamic_params.get('temperature', 0.75),
                    "top_p": dynamic_params.get('top_p', 0.9),
                    "max_tokens": dynamic_params.get('max_tokens', 1012),
                    **generation_params
                }
                
                print(f"Together API request (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    url, headers=headers, json=payload,
                    timeout=60, verify=certifi.where()
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
                
            except requests.exceptions.Timeout:
                print(f"Together API timeout (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError as e:
                print(f"Together API connection error (attempt {attempt + 1}/{max_retries}): {e}")
            except requests.exceptions.RequestException as e:
                print(f"Together API request error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                print(f"Together API error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(retry_delay)
        
        print("Together API: all retries failed")
        return ""
    
    def call_qwen_api(
        self,
        messages: List[Dict],
        agent_id: str = "",
        math_problem: str = "",
        position: int = 0,
        previous_solutions: Optional[List[str]] = None,
        problem_index: int = 0,
        **generation_params
    ) -> str:
        """
        Call Alibaba DashScope API for Qwen model generation.
        
        Uses streaming mode with thinking capability enabled.
        
        Args:
            messages: Chat messages in OpenAI format.
            agent_id: Agent identifier for dynamic parameters.
            math_problem: Problem text for state encoding.
            position: Position in sequential game.
            previous_solutions: Previous agent solutions.
            problem_index: Problem index for logging.
            **generation_params: Additional generation parameters.
        
        Returns:
            Generated response text, or empty string on failure.
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                agent_id_for_params = agent_id if agent_id else 'Agent_Qwen'
                dynamic_params = self.generate_dynamic_params(
                    agent_id_for_params, math_problem, position, previous_solutions, problem_index
                )
                
                payload = {
                    "model": "qwen3-8b",
                    "messages": messages,
                    "temperature": dynamic_params.get('temperature', 0.6),
                    "max_tokens": dynamic_params.get('max_tokens', 1012),
                    "top_p": dynamic_params.get('top_p', 0.95),
                    "stream": True,
                    "extra_body": {"enable_thinking": True}
                }
                
                print(f"DashScope API request (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    url, headers=headers, json=payload,
                    timeout=60, verify=certifi.where(), stream=True
                )
                
                if response.status_code == 200:
                    reasoning_content = ""
                    answer_content = ""
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and chunk['choices']:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'reasoning_content' in delta and delta['reasoning_content']:
                                            reasoning_content += delta['reasoning_content']
                                        if 'content' in delta and delta['content']:
                                            answer_content += delta['content']
                                except json.JSONDecodeError:
                                    continue
                    
                    if reasoning_content:
                        print(f"\n=== Qwen Thinking Process ===")
                        print(reasoning_content)
                        print(f"=== End Thinking ===\n")
                    
                    final_response = answer_content if answer_content else reasoning_content
                    return final_response.strip() if final_response else ""
                else:
                    try:
                        error_info = response.json()
                        print(f"API error details: {json.dumps(error_info, ensure_ascii=False)}")
                    except Exception:
                        pass
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                time.sleep(retry_delay)
        
        print("DashScope API: all retries failed")
        return ""
    
    def load_local_model(self, model_key: str) -> bool:
        """
        Load a local HuggingFace model.
        
        Args:
            model_key: Key identifying the model (e.g., "SMOLLM2").
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if model_key in self.local_models:
            return True
        
        try:
            if model_key == "SMOLLM2":
                model_name = SMOLLM2_CONFIG["name"]
                torch_dtype = SMOLLM2_CONFIG["torch_dtype"]
                
                print(f"Loading SmolLM2 model: {model_name}")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, use_fast=False
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                self.local_models[model_key] = {"model": model, "tokenizer": tokenizer}
                print(f"SmolLM2 model loaded successfully")
                return True
                
        except Exception as e:
            print(f"Error loading SmolLM2 model: {e}")
            return False
    
    def generate_local_response(
        self,
        model_key: str,
        messages: List[Dict],
        agent_id: str = "",
        math_problem: str = "",
        position: int = 0,
        previous_solutions: Optional[List[str]] = None,
        problem_index: int = 0,
        **generation_params
    ) -> str:
        """
        Generate response using a local model.
        
        Args:
            model_key: Key identifying the local model.
            messages: Chat messages.
            agent_id: Agent identifier for dynamic parameters.
            math_problem: Problem text for state encoding.
            position: Position in sequential game.
            previous_solutions: Previous agent solutions.
            problem_index: Problem index for logging.
            **generation_params: Additional generation parameters.
        
        Returns:
            Generated response text, or empty string on failure.
        """
        if model_key not in self.local_models:
            return ""
        
        try:
            model_data = self.local_models[model_key]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            agent_id_for_params = agent_id if agent_id else 'Agent_SMOLLM2'
            dynamic_params = self.generate_dynamic_params(
                agent_id_for_params, math_problem, position, previous_solutions, problem_index
            )
            
            if not dynamic_params:
                dynamic_params = {
                    'temperature': 0.8, 'top_p': 0.85,
                    'max_tokens': 256, 'repetition_penalty': 1.15
                }
            
            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                input_text = f"User: {messages[-1]['content']}\nAssistant:"
            
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(autocast_device):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=dynamic_params.get('max_tokens', 256),
                    temperature=dynamic_params.get('temperature', 0.8),
                    top_p=dynamic_params.get('top_p', 0.85),
                    repetition_penalty=dynamic_params.get('repetition_penalty', 1.15),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_params
                )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            return response.strip()
            
        except torch.cuda.OutOfMemoryError:
            try:
                model.to('cpu')
                inputs = tokenizer(input_text, return_tensors="pt").to('cpu')
                with torch.amp.autocast('cpu'):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(dynamic_params.get('max_tokens', 256), 256),
                        temperature=dynamic_params.get('temperature', 0.8),
                        top_p=dynamic_params.get('top_p', 0.85),
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                return response.strip()
            except Exception as cpu_e:
                print(f"CPU fallback also failed: {cpu_e}")
                return ""
        except Exception as e:
            print(f"Local model generation failed: {e}")
            return ""


class MathProblemPool:
    """
    Pool for collecting and managing solutions in the public goods game.
    
    Stores individual agent solutions, tracks contribution history,
    and provides answer extraction utilities.
    
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
        """
        Add an individual agent's solution to the pool.
        
        Args:
            agent_id: Identifier of the contributing agent.
            solution: The solution text.
            problem_index: Index of the problem (for logging).
        """
        self.individual_solutions[agent_id] = solution
        self.contribution_history.append({
            "agent_id": agent_id,
            "timestamp": time.time(),
            "solution_type": "individual",
            "content": solution
        })
    
    def set_final_solution(
        self,
        solution: str,
        integrator_id: str,
        problem_index: int = 0
    ) -> None:
        """
        Set the final integrated solution.
        
        Args:
            solution: The final solution text.
            integrator_id: Identifier of the integrating agent.
            problem_index: Index of the problem (for logging).
        """
        self.final_solution = solution
        self.contribution_history.append({
            "agent_id": integrator_id,
            "timestamp": time.time(),
            "solution_type": "final_integration",
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
        """
        Extract numerical answer from solution text.
        
        Delegates to the utility function for answer extraction.
        
        Args:
            text: Solution text to extract answer from.
            show_debug: Whether to print debug information.
        
        Returns:
            Extracted answer string.
        """
        return extract_numerical_answer(text, show_debug)


class MathAgent:
    """
    Agent for mathematical reasoning in the SPGG framework.
    
    Generates solutions to math problems using various LLM backends
    (Together API, DashScope API, or local models).
    
    Attributes:
        agent_id: Unique identifier for the agent.
        model_type: Type of model backend ("together", "qwen", "local").
        api_manager: APIModelManager instance for LLM calls.
    """
    
    def __init__(self, agent_id: str, model_type: str, api_manager: APIModelManager):
        """
        Initialize a math agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            model_type: Type of model backend.
            api_manager: APIModelManager instance.
        """
        self.agent_id = agent_id
        self.model_type = model_type
        self.api_manager = api_manager
    
    def generate_solution(
        self,
        problem: str,
        previous_solution: Optional[Dict[str, str]] = None,
        position: int = 0,
        problem_index: int = 0,
        is_last_agent: bool = False
    ) -> str:
        """
        Generate a solution for the given math problem.
        
        Args:
            problem: The mathematical problem text.
            previous_solution: Solution from the previous agent (partial observation).
            position: Position in the sequential game.
            problem_index: Index of current problem.
            is_last_agent: Whether this is the final agent in the sequence.
        
        Returns:
            Generated solution text.
        """
        prompt = self._build_prompt(problem, previous_solution, is_last_agent)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        previous_solution_texts = [previous_solution["solution"]] if previous_solution else []
        
        try:
            if self.model_type == "together":
                return self.api_manager.call_together_api(
                    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    messages,
                    agent_id=self.agent_id,
                    math_problem=problem,
                    position=position,
                    previous_solutions=previous_solution_texts,
                    problem_index=problem_index
                )
            elif self.model_type == "qwen":
                return self.api_manager.call_qwen_api(
                    messages,
                    agent_id=self.agent_id,
                    math_problem=problem,
                    position=position,
                    previous_solutions=previous_solution_texts,
                    problem_index=problem_index
                )
            elif self.model_type == "local":
                return self.api_manager.generate_local_response(
                    "SMOLLM2",
                    messages,
                    agent_id=self.agent_id,
                    math_problem=problem,
                    position=position,
                    previous_solutions=previous_solution_texts,
                    problem_index=problem_index
                )
            else:
                return ""
        except Exception as e:
            print(f"{self.agent_id} solution generation failed: {e}")
            return ""
    
    def _build_prompt(
        self,
        problem: str,
        previous_solution: Optional[Dict[str, str]] = None,
        is_last_agent: bool = False
    ) -> str:
        """
        Build the prompt for the agent.
        
        Args:
            problem: The mathematical problem.
            previous_solution: Previous agent's solution (partial observation).
            is_last_agent: Whether this is the final agent.
        
        Returns:
            Formatted prompt string.
        """
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
