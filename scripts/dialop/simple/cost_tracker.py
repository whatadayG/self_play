import json
import os
import glob
from datetime import datetime
import tiktoken
from typing import Dict, List, Optional, Union
try:
    from gpt_cost_estimator import CostEstimator  # type: ignore
except Exception:
    class CostEstimator:  # type: ignore
        @staticmethod
        def estimate(model: str, input_tokens: int, output_tokens: int) -> float:
            return 0.0

class CostTracker:
    """
    Utility to track and summarize OpenAI API costs across multiple experiments.
    """
    # Current pricing per 1M tokens (as of December 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40},
        "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
        "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
        "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
        "gpt-4": {"input": 30.00, "output": 60.00},  # Keeping older model pricing for backward compatibility
        "gpt-4-32k": {"input": 60.00, "output": 120.00},  # Keeping older model pricing for backward compatibility
        "gpt-3.5-turbo": {"input": 1.50, "output": 2.00},  # Keeping older model pricing for backward compatibility
        "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},  
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "o3-2025-04-16": {"input": 1.00, "output": 4.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},

        # Add other models as needed
    }
    
    def __init__(self, results_dir: str):
        """
        Initialize the cost tracker.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.summary_file = os.path.join(results_dir, "cost_summary.json")
    
    def _get_api_log_files(self) -> List[str]:
        """Get all API log files in the results directory and subdirectories."""
        return glob.glob(os.path.join(self.results_dir, "**", "*_api.json"), recursive=True)
    
    def calculate_cost(self, api_calls: List[Dict]) -> Dict:
        """
        Calculate cost for a list of API calls.
        
        Args:
            api_calls: List of API call records
            
        Returns:
            Dictionary with cost breakdown
        """
        total_cost = 0.0
        model_costs = {}
        total_tokens = {"input": 0, "output": 0}
        
        for call in api_calls:
            model = call.get("model", "unknown")
            input_tokens = call.get("input_tokens", 0)
            output_tokens = call.get("output_tokens", 0)
            
            # Initialize model in tracking dictionaries if not present
            if model not in model_costs:
                model_costs[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            
            # Update token counts
            model_costs[model]["calls"] += 1
            model_costs[model]["input_tokens"] += input_tokens
            model_costs[model]["output_tokens"] += output_tokens
            total_tokens["input"] += input_tokens
            total_tokens["output"] += output_tokens
            
            # Calculate cost if pricing is available
            if model in self.PRICING:
                # Convert to millions of tokens for pricing calculation
                input_cost = (input_tokens / 1000000) * self.PRICING[model]["input"]
                output_cost = (output_tokens / 1000000) * self.PRICING[model]["output"]
                call_cost = input_cost + output_cost
                
                model_costs[model]["cost"] += call_cost
                total_cost += call_cost
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "model_breakdown": model_costs,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_experiment(self, experiment_path: str) -> Dict:
        """
        Analyze costs for a specific experiment.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Cost summary for the experiment
        """
        api_files = glob.glob(os.path.join(experiment_path, "*_api.json"))
        all_calls = []
        
        for file_path in api_files:
            try:
                with open(file_path, 'r') as f:
                    calls = json.load(f)
                    if isinstance(calls, list):
                        all_calls.extend(calls)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return self.calculate_cost(all_calls)
    
    def generate_summary(self, save: bool = True) -> Dict:
        """
        Generate a cost summary for all experiments.
        
        Args:
            save: Whether to save the summary to a file
            
        Returns:
            Complete cost summary
        """
        api_files = self._get_api_log_files()
        all_calls = []
        experiment_summaries = {}
        files_processed = 0  # Counter for processed files
        
        # Group files by experiment
        for file_path in api_files:
            # Extract experiment name from path
            rel_path = os.path.relpath(file_path, self.results_dir)
            parts = rel_path.split(os.sep)
            
            if len(parts) > 1:
                experiment = parts[0]  # First directory is experiment name
            else:
                experiment = "default"
                
            if experiment not in experiment_summaries:
                experiment_summaries[experiment] = []
                
            try:
                with open(file_path, 'r') as f:
                    calls = json.load(f)
                    if isinstance(calls, list):
                        experiment_summaries[experiment].extend(calls)
                        all_calls.extend(calls)
                        files_processed += 1  # Increment counter on successful processing
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Calculate costs for each experiment
        experiment_costs = {}
        for experiment, calls in experiment_summaries.items():
            experiment_costs[experiment] = self.calculate_cost(calls)
        
        # Calculate overall costs
        overall_summary = self.calculate_cost(all_calls)
        
        # Prepare final summary
        summary = {
            "overall": overall_summary,
            "experiments": experiment_costs,
            "files_processed": files_processed,  # Add files processed count to summary
            "total_files_found": len(api_files),  # Add total files found
            "generated_at": datetime.now().isoformat()
        }
        
        if save:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Cost summary saved to {self.summary_file}")
        
        return summary
    
    def print_summary(self, detailed: bool = False):
        """
        Print a human-readable cost summary.
        
        Args:
            detailed: Whether to include detailed breakdown by model
        """
        summary = self.generate_summary(save=True)
        
        print("\n===== OPENAI API COST SUMMARY =====")
        print(f"Generated at: {summary['generated_at']}")
        print(f"Files processed: {summary['files_processed']}/{summary['total_files_found']} API log files")
        print(f"Total cost: ${summary['overall']['total_cost']:.2f}")
        print(f"Total tokens: {summary['overall']['total_tokens']['input'] + summary['overall']['total_tokens']['output']:,} " +
              f"(Input: {summary['overall']['total_tokens']['input']:,}, Output: {summary['overall']['total_tokens']['output']:,})")
        
        if detailed:
            print("\n----- Model Breakdown -----")
            for model, data in summary['overall']['model_breakdown'].items():
                print(f"\n{model}:")
                print(f"  Calls: {data['calls']}")
                print(f"  Tokens: {data['input_tokens'] + data['output_tokens']:,} " +
                      f"(Input: {data['input_tokens']:,}, Output: {data['output_tokens']:,})")
                print(f"  Cost: ${data['cost']:.2f}")
        
        print("\n----- Experiment Breakdown -----")
        for experiment, data in summary['experiments'].items():
            print(f"\n{experiment}:")
            print(f"  Cost: ${data['total_cost']:.2f}")
            print(f"  Tokens: {data['total_tokens']['input'] + data['total_tokens']['output']:,}")
            
            if detailed:
                print("  Models:")
                for model, model_data in data['model_breakdown'].items():
                    print(f"    {model}: ${model_data['cost']:.2f} ({model_data['calls']} calls)")
        
        print("\n=====================================")


class ConversationCostTracker:
    """
    A cost tracking utility that integrates with gpt-cost-estimator
    to track OpenAI API costs across conversations.
    """
    
    def __init__(self, custom_prices: Optional[Dict] = None):
        """
        Initialize the cost tracker.
        
        Args:
            custom_prices: Optional dictionary of custom model prices
                          Format: {"model-name": {"input": price_per_token, "output": price_per_token}}
        """
        # Custom prices for newer models if needed
        default_custom_prices = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        }
        
        if custom_prices:
            default_custom_prices.update(custom_prices)
            
        self.cost_estimator = CostEstimator(price_overrides=default_custom_prices)
        self.conversation_costs = []
        self.current_conversation_id = None
        
    def start_conversation(self, conversation_id: str):
        """Start tracking a new conversation."""
        self.current_conversation_id = conversation_id
        CostEstimator.reset()  # Reset the global cost counter
        
    def end_conversation(self) -> Dict:
        """End the current conversation and return cost summary."""
        if self.current_conversation_id is None:
            return {"error": "No active conversation"}
            
        total_cost = CostEstimator.total_cost
        conversation_summary = {
            "conversation_id": self.current_conversation_id,
            "total_cost": total_cost,
            "end_time": datetime.now().isoformat(),
            "cost_breakdown": self.get_detailed_cost_breakdown()
        }
        
        self.conversation_costs.append(conversation_summary)
        self.current_conversation_id = None
        return conversation_summary
        
    def get_current_cost(self) -> float:
        """Get the current conversation cost."""
        return CostEstimator.total_cost
        
    def get_detailed_cost_breakdown(self) -> Dict:
        """Get detailed cost breakdown for the current conversation."""
        # This is a simplified version - you could extend this to track
        # costs per player, per model, etc.
        return {
            "total_cost": CostEstimator.total_cost,
            "currency": "USD"
        }
        
    def get_all_conversations_summary(self) -> Dict:
        """Get summary of all tracked conversations."""
        total_cost = sum(conv["total_cost"] for conv in self.conversation_costs)
        return {
            "total_conversations": len(self.conversation_costs),
            "total_cost_all_conversations": total_cost,
            "conversations": self.conversation_costs
        }
        
    def save_cost_report(self, filename: str):
        """Save cost report to a JSON file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_all_conversations_summary(),
            "conversations": self.conversation_costs
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
    def create_decorated_api_function(self):
        """
        Create a decorated OpenAI API function for cost tracking.
        Returns a function that can be used to make tracked API calls.
        """
        @self.cost_estimator
        def tracked_openai_call(client, model, messages, **kwargs):
            """
            Make a tracked OpenAI API call.
            
            Args:
                client: OpenAI client instance
                model: Model name
                messages: List of messages
                **kwargs: Additional API parameters
            """
            # Remove cost estimator specific arguments
            args_to_remove = ['mock', 'completion_tokens']
            for arg in args_to_remove:
                if arg in kwargs:
                    del kwargs[arg]
                    
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
        return tracked_openai_call
        
    def print_cost_summary(self, conversation_id: str = None):
        """Print a formatted cost summary."""
        if conversation_id:
            # Print summary for specific conversation
            conv = next((c for c in self.conversation_costs if c["conversation_id"] == conversation_id), None)
            if conv:
                print(f"\n{'='*50}")
                print(f"COST SUMMARY - Conversation: {conversation_id}")
                print(f"{'='*50}")
                print(f"Total Cost: ${conv['total_cost']:.6f}")
                print(f"End Time: {conv['end_time']}")
                print(f"{'='*50}\n")
            else:
                print(f"Conversation '{conversation_id}' not found.")
        else:
            # Print current conversation summary
            current_cost = CostEstimator.total_cost
            print(f"\n{'='*50}")
            print(f"CURRENT CONVERSATION COST")
            print(f"{'='*50}")
            print(f"Current Cost: ${current_cost:.6f}")
            if self.current_conversation_id:
                print(f"Conversation ID: {self.current_conversation_id}")
            print(f"{'='*50}\n")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Track OpenAI API costs across experiments")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown by model")
    
    args = parser.parse_args()
    
    tracker = CostTracker(args.results_dir)
    tracker.print_summary(detailed=args.detailed) 