import json
import os
from datetime import datetime
from typing import Dict, Optional
import logging
from dataclasses import dataclass, asdict
import tiktoken

@dataclass
class APICall:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    project_id: str
    cost: float
    endpoint: str

class OpenAITracker:
    # Current pricing as of 2024 (per 1M tokens)
    PRICING = {
        'gpt-4': {'input': 30.00, 'output': 60.00},
        'gpt-4-0613': {'input': 30.00, 'output': 60.00},
        'gpt-4-32k': {'input': 60.00, 'output': 120.00},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00},
        'chatgpt-4o-latest': {'input': 5.00, 'output': 15.00},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-2024-08-06': {'input': 2.50, 'output': 10.00},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        'gpt-3.5-turbo-0125': {'input': 0.50, 'output': 1.50},
        'gpt-3.5-turbo-instruct': {'input': 1.50, 'output': 2.00},
        'gpt-3.5-turbo-16k-0613': {'input': 3.00, 'output': 4.00},
        'davinci-002': {'input': 2.00, 'output': 2.00},
        'babbage-002': {'input': 0.40, 'output': 0.40},
        "gpt-4.1": {"input": 2, "output": 8},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "o4-mini": {"input": 1.1, "output": 4.4},
        "o3-2025-04-16": {"input": 2.00, "output": 8.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
    }
    
    def __init__(self, log_file: str = 'openai_usage.json'):
        self.log_file = log_file
        self.api_calls = []  # Store calls in memory
        self.logger = logging.getLogger('OpenAITracker')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler('openai_tracker.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text using tiktoken.

        The newest OpenAI model names (e.g., "gpt-4.1") are sometimes not yet
        recognised by `tiktoken`.  When that happens we fall back to the
        `cl100k_base` encoding, which is known to be compatible with all GPT-4
        and GPT-3.5 style chat models.  This prevents the silent failure that
        previously returned 0 for both input and output tokens.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model name unknown to tiktoken – use the most compatible base
            # encoding instead of failing hard.
            self.logger.warning(
                f"Model '{model}' not found in tiktoken mapping; falling back to cl100k_base encoding for token count.")
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # Any other unexpected error: log and fall back to base encoding.
            self.logger.error(f"Unexpected error obtaining encoding for model '{model}': {e}. Using cl100k_base.")
            encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error encoding text for model '{model}': {e}")
            return 0

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call based on model and token counts"""
        if model not in self.PRICING:
            self.logger.warning(f"Unknown model {model}, cost calculation may be inaccurate")
            return 0.0
        
        pricing = self.PRICING[model]
        # Convert from price per 1M tokens to price per token
        input_cost = (input_tokens / 1000000) * pricing['input']
        output_cost = (output_tokens / 1000000) * pricing['output']
        return input_cost + output_cost

    def log_api_call(self, 
                     model: str,
                     input_text: str,
                     output_text: str,
                     project_id: str,
                     endpoint: str = "completions") -> None:
        """Log an API call with its associated costs and metadata"""
        try:
            input_tokens = self.count_tokens(input_text, model)
            output_tokens = self.count_tokens(output_text, model)
            cost = self.calculate_cost(model, input_tokens, output_tokens)
            
            api_call = APICall(
                timestamp=datetime.now().isoformat(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                project_id=project_id,
                cost=cost,
                endpoint=endpoint
            )
            
            self._save_to_log(api_call)
            self.logger.info(f"Logged API call for project {project_id}: {cost:.4f} USD")
            
        except Exception as e:
            self.logger.error(f"Error logging API call: {e}")

    def _save_to_log(self, api_call: APICall) -> None:
        """Save API call to memory and optionally to file"""
        try:
            self.api_calls.append(asdict(api_call))
            
            # Also save to file if specified
            if self.log_file:
                try:
                    # Read existing data if file exists
                    if os.path.exists(self.log_file):
                        with open(self.log_file, 'r') as f:
                            existing_data = json.load(f)
                        # Append new call to existing data
                        existing_data.append(asdict(api_call))
                        with open(self.log_file, 'w') as f:
                            json.dump(existing_data, f, indent=2)
                    else:
                        # Create new file with just this call
                        with open(self.log_file, 'w') as f:
                            json.dump([asdict(api_call)], f, indent=2)
                except Exception as e:
                    self.logger.error(f"Error reading/writing to log file: {e}")
                    # Fallback to overwrite mode
                    with open(self.log_file, 'w') as f:
                        json.dump(self.api_calls, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving to log: {e}")

    def get_project_costs(self, project_id: Optional[str] = None) -> Dict:
        """Get usage statistics and costs for a specific project or all projects"""
        try:
            logs = self.api_calls
            
            if project_id:
                logs = [log for log in logs if log['project_id'] == project_id]
            
            stats = {
                'total_cost': sum(log['cost'] for log in logs),
                'total_input_tokens': sum(log['input_tokens'] for log in logs),
                'total_output_tokens': sum(log['output_tokens'] for log in logs),
                'call_count': len(logs),
                'models_used': list(set(log['model'] for log in logs))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting project costs: {e}")
            return {}

    def get_api_call_history(self, project_id: Optional[str] = None) -> list:
        """Get the full history of API calls for a project"""
        if project_id:
            return [call for call in self.api_calls if call['project_id'] == project_id]
        return self.api_calls.copy()