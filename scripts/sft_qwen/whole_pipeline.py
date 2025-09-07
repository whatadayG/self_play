import os
import argparse
import json
import time
from pathlib import Path
from openai import OpenAI
import random
import numpy as np
from tqdm import tqdm
import sys
import pdb  # Add pdb import
import subprocess
import ast
import shlex
import multiprocessing
from pathlib import Path
import time
from typing import Optional, Dict, Any
import tyro
# Will import conditionally based on persona parameter
import concurrent.futures

# Add the project root to the path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use direct imports with absolute paths to avoid package import issues
# from evaluate_checkpoints import evaluate_latest_job  # Commented out
from eval_cp_experiment import evaluate_checkpoints  # Added new import
from dialop.c_run_multiple import parallel_conversations as complex_parallel_conversations
from dialop.notebooks.simple.s_run_multiple import parallel_conversations as simple_parallel_conversations
from simple_data import format_for_agent_model, format_for_user_model
from optimization_data import format_for_agent_model as format_for_agent_model_optimization
from optimization_data import format_for_user_model as format_for_user_model_optimization
from finetune_parallel import create_training_files, wait_for_file_processing, create_fine_tuning_job, shuffle_dataset, prepare_data_split

# Import evaluate_opt for optimization mode
from dialop.evaluate_opt import main as evaluate_opt_main

def run_single_checkpoint_evaluation(checkpoint_info: Dict[str, Any], eval_args, exp_name_base: str):
    """Run evaluation for a single checkpoint"""
    try:
        # Delayed imports to avoid heavyweight modules when not needed
        from openai import OpenAI
        from dialop.results.matching.normalized_reward_extraction import process_directory
        import dialop.evaluate_opt as eval_opt_mod
        from evaluate_opt import main as evaluate_opt_main

        checkpoint_id = checkpoint_info.get("checkpoint_id")
        checkpoint_index = checkpoint_info.get("index")
        
        print(f"Running evaluation for checkpoint {checkpoint_index}: {checkpoint_id}")
        start = time.time()

        # Unique experiment name so that result directories do not clash
        timestamp = int(time.time()) + checkpoint_index  # Add index to avoid timestamp collision
        eval_exp_name = f"{exp_name_base}_eval_{checkpoint_id.replace(':', '_')}_{timestamp}"

        # ------------------------------------------------------------------
        # Run evaluation with evaluate_opt.main (10 conversations)
        # ------------------------------------------------------------------
        try:
            # Decide which side the checkpoint will play
            if getattr(eval_args, "eval_side", "player2") == "player2":
                user_model_id = checkpoint_id  # checkpoint plays as player-2
                agent_model_id = "gpt-4.1"    # baseline on player-1
            else:
                user_model_id = "gpt-4.1"     # baseline on player-2
                agent_model_id = checkpoint_id  # checkpoint plays as player-1

            evaluate_opt_main(
                exp_name=eval_exp_name,
                game="matching",
                mode="selfplay",
                new_data=True,
                resume=0,
                end=10,  # Exactly 10 conversations as requested
                samples_per_game=1,
                user_model_id=user_model_id,
                agent_model_id=agent_model_id,
                dry_run=False,
                use_word_limit=False,
                track_costs=False,
                threshold=0.5
            )
        except Exception as run_err:
            print(f"❌  Evaluation run failed for {checkpoint_id}: {run_err}")
            return {
                "checkpoint_id": checkpoint_id,
                "success": False,
                "error": str(run_err),
                "avg_score": 0.0,
                "exp_name": eval_exp_name
            }

        # ------------------------------------------------------------------
        # Parse evaluation outputs to obtain average score_norm
        # ------------------------------------------------------------------
        result_dir = eval_opt_mod.RESDIR / "matching" / eval_exp_name

        if not result_dir.exists():
            print(f"⚠️  Result directory not found for {checkpoint_id}: {result_dir}")
            return {
                "checkpoint_id": checkpoint_id,
                "success": False,
                "error": "Result directory missing",
                "avg_score": 0.0,
                "exp_name": eval_exp_name
            }

        score_norms, _, _, _, _ = process_directory(result_dir)

        if not score_norms:
            print(f"⚠️  No valid score_norm values extracted for {checkpoint_id} – treating as 0.0")
            avg_score = 0.0
        else:
            avg_score = float(sum(score_norms) / len(score_norms))

        elapsed = (time.time() - start) / 60
        print(f"✅  Completed evaluation for {checkpoint_id}. Average score_norm: {avg_score:.4f} in {elapsed:.1f} minutes")

        return {
            "checkpoint_id": checkpoint_id,
            "success": True,
            "error": None,
            "avg_score": avg_score,
            "exp_name": eval_exp_name,
            "scores": score_norms,
            "elapsed_minutes": elapsed
        }

    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_id}: {str(e)}")
        return {
            "checkpoint_id": checkpoint_id,
            "success": False,
            "error": str(e),
            "avg_score": 0.0,
            "exp_name": eval_exp_name
        }

def evaluate_optimization_checkpoints_parallel(eval_args, timeout_minutes: int = 30):
    """
    Evaluate all checkpoints of a fine-tuning job in parallel (assumed to be three) by running
    `evaluate_opt.py` for 10 conversations each and computing the average
    `score_norm` for every checkpoint.

    Args:
        eval_args: Evaluation arguments
        timeout_minutes: Maximum time to wait for each checkpoint evaluation
    
    Returns:
        Dictionary with win_percentages and details for each checkpoint
    """
    print(f"Running parallel optimization evaluation for job_id: {eval_args.job_id}")

    try:
        # Delayed imports to avoid heavyweight modules when not needed
        from openai import OpenAI
        from list_checkpoints import get_latest_job_checkpoints

        client = OpenAI()

        # ------------------------------------------------------------------
        # 1. Retrieve the checkpoints for the fine-tuning job
        # ------------------------------------------------------------------
        job_info = get_latest_job_checkpoints(client, eval_args.job_id, verbose=False)

        if not job_info or not job_info.get("checkpoints"):
            print("No checkpoints returned from OpenAI API – aborting evaluation.")
            return None

        checkpoints = job_info["checkpoints"][:3]  # first three
        print(f"Found {len(checkpoints)} checkpoints to evaluate in parallel")

        # Prepare checkpoint info for parallel processing
        checkpoint_tasks = []
        for idx, checkpoint in enumerate(checkpoints, 1):
            # Handle two possible representations: dict (from helper) or SDK object
            if isinstance(checkpoint, dict):
                checkpoint_id = checkpoint.get("model_id") or checkpoint.get("checkpoint_id")
            else:
                checkpoint_id = (
                    getattr(checkpoint, "fine_tuned_model_checkpoint", None)
                    or getattr(checkpoint, "fine_tuned_model", None)
                    or getattr(checkpoint, "id", None)
                )

            if checkpoint_id is None:
                print(f"Warning: Could not determine model id for checkpoint {checkpoint} – skipping.")
                continue

            checkpoint_tasks.append({
                "checkpoint_id": checkpoint_id,
                "index": idx
            })

        if not checkpoint_tasks:
            print("No valid checkpoints found for evaluation.")
            return None

        # ------------------------------------------------------------------
        # 2. Run parallel evaluation
        # ------------------------------------------------------------------
        evaluation_results = {
            "win_percentages": {},
            "details": {}
        }

        # Use ProcessPoolExecutor for parallel execution
        max_workers = min(len(checkpoint_tasks), 3)  # Max 3 workers for 3 checkpoints
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all checkpoint evaluation tasks
            future_to_checkpoint = {
                executor.submit(run_single_checkpoint_evaluation, checkpoint_info, eval_args, eval_args.exp_name): checkpoint_info["checkpoint_id"]
                for checkpoint_info in checkpoint_tasks
            }
            
            # Handle results as they complete
            for future in concurrent.futures.as_completed(future_to_checkpoint):
                checkpoint_id = future_to_checkpoint[future]
                try:
                    result = future.result(timeout=timeout_minutes * 60)  # Convert to seconds
                    
                    if result["success"]:
                        evaluation_results["win_percentages"][checkpoint_id] = result["avg_score"]
                        evaluation_results["details"][checkpoint_id] = {
                            "exp_name": result["exp_name"],
                            "scores": result.get("scores", []),
                            "average_score_norm": result["avg_score"],
                            "elapsed_minutes": result.get("elapsed_minutes", 0)
                        }
                    else:
                        evaluation_results["win_percentages"][checkpoint_id] = 0.0
                        evaluation_results["details"][checkpoint_id] = {
                            "error": result["error"]
                        }
                        
                except concurrent.futures.TimeoutError:
                    print(f"Checkpoint evaluation {checkpoint_id} timed out after {timeout_minutes} minutes!")
                    evaluation_results["win_percentages"][checkpoint_id] = 0.0
                    evaluation_results["details"][checkpoint_id] = {
                        "error": f"Timed out after {timeout_minutes} minutes"
                    }
                    future.cancel()
                except Exception as e:
                    print(f"Checkpoint evaluation {checkpoint_id} generated an exception: {e}")
                    evaluation_results["win_percentages"][checkpoint_id] = 0.0
                    evaluation_results["details"][checkpoint_id] = {
                        "error": str(e)
                    }
                    future.cancel()
            
            # Ensure all processes are properly shut down
            executor.shutdown(wait=True)

        # ------------------------------------------------------------------
        # 3. Report results
        # ------------------------------------------------------------------
        completed_evals = [k for k, v in evaluation_results["win_percentages"].items() if v > 0]
        total_evals = len(checkpoint_tasks)
        
        print(f"\nParallel evaluation completed:")
        print(f"✅ Successful evaluations: {len(completed_evals)}/{total_evals}")
        
        for checkpoint_id, score in evaluation_results["win_percentages"].items():
            status = "✅" if score > 0 else "❌"
            print(f"  {status} {checkpoint_id}: {score:.4f}")

        return evaluation_results

    except Exception as e:
        print(f"Error during parallel optimization evaluation setup: {e}")
        return None

def evaluate_optimization_checkpoints(eval_args):
    """
    Sequential version: Evaluate all checkpoints of a fine-tuning job (assumed to be three) by running
    `evaluate_opt.py` for 10 conversations each and computing the average
    `score_norm` for every checkpoint.

    The function returns a dictionary with the same structure expected by
    `find_best_checkpoint`, i.e. a top-level key `win_percentages` mapping the
    checkpoint-model-id to its average normalised reward.  Additional
    information is stored under the `details` key for debugging.
    """

    print(f"Running optimization evaluation for job_id: {eval_args.job_id}")

    try:
        # Delayed imports to avoid heavyweight modules when not needed
        from openai import OpenAI
        from dialop.results.matching.normalized_reward_extraction import process_directory
        import dialop.evaluate_opt as eval_opt_mod
        from list_checkpoints import get_latest_job_checkpoints

        client = OpenAI()

        # ------------------------------------------------------------------
        # 1. Retrieve the checkpoints for the fine-tuning job
        # ------------------------------------------------------------------
        # Use the shared helper that already handles checkpoint retrieval
        job_info = get_latest_job_checkpoints(client, eval_args.job_id, verbose=False)

        if not job_info or not job_info.get("checkpoints"):
            print("No checkpoints returned from OpenAI API – aborting evaluation.")
            return None

        checkpoints = job_info["checkpoints"][:3]  # first three

        # ------------------------------------------------------------------
        # 2. Iterate over checkpoints, run evaluation, and collect scores
        # ------------------------------------------------------------------
        evaluation_results = {
            "win_percentages": {},  # key expected by downstream logic
            "details": {}
        }

        for idx, checkpoint in enumerate(checkpoints, 1):
            # Handle two possible representations: dict (from helper) or SDK object
            if isinstance(checkpoint, dict):
                checkpoint_id = checkpoint.get("model_id") or checkpoint.get("checkpoint_id")
            else:
                checkpoint_id = (
                    getattr(checkpoint, "fine_tuned_model_checkpoint", None)
                    or getattr(checkpoint, "fine_tuned_model", None)
                    or getattr(checkpoint, "id", None)
                )

            if checkpoint_id is None:
                print(f"Warning: Could not determine model id for checkpoint {checkpoint} – skipping.")
                continue

            print(f"\n--- [{idx}/{len(checkpoints)}] Evaluating checkpoint: {checkpoint_id}")

            # Unique experiment name so that result directories do not clash
            timestamp = int(time.time())
            eval_exp_name = f"{eval_args.exp_name}_eval_{checkpoint_id.replace(':', '_')}_{timestamp}"

            # ------------------------------------------------------------------
            # Run evaluation with evaluate_opt.main (10 conversations)
            # ------------------------------------------------------------------
            try:
                # Decide which side the checkpoint will play
                if getattr(eval_args, "eval_side", "player2") == "player2":
                    user_model_id = checkpoint_id  # checkpoint plays as player-2
                    agent_model_id = "gpt-4.1"    # baseline on player-1
                else:
                    user_model_id = "gpt-4.1"     # baseline on player-2
                    agent_model_id = checkpoint_id  # checkpoint plays as player-1

                evaluate_opt_main(
                    exp_name=eval_exp_name,
                    game="matching",
                    mode="selfplay",
                    new_data=True,
                    resume=0,
                    end=10,  # Exactly 10 conversations as requested
                    samples_per_game=1,
                    user_model_id=user_model_id,
                    agent_model_id=agent_model_id,
                    dry_run=False,
                    use_word_limit=False,
                    track_costs=False,
                    threshold=0.5
                )
            except Exception as run_err:
                print(f"❌  Evaluation run failed for {checkpoint_id}: {run_err}")
                evaluation_results["win_percentages"][checkpoint_id] = 0.0
                evaluation_results["details"][checkpoint_id] = {
                    "error": str(run_err)
                }
                continue

            # ------------------------------------------------------------------
            # 3. Parse evaluation outputs to obtain average score_norm
            # ------------------------------------------------------------------
            result_dir = eval_opt_mod.RESDIR / "matching" / eval_exp_name

            if not result_dir.exists():
                print(f"⚠️  Result directory not found for {checkpoint_id}: {result_dir}")
                evaluation_results["win_percentages"][checkpoint_id] = 0.0
                evaluation_results["details"][checkpoint_id] = {
                    "error": "Result directory missing"
                }
                continue

            score_norms, _, _, _, _ = process_directory(result_dir)

            if not score_norms:
                print(f"⚠️  No valid score_norm values extracted for {checkpoint_id} – treating as 0.0")
                avg_score = 0.0
            else:
                avg_score = float(sum(score_norms) / len(score_norms))

            evaluation_results["win_percentages"][checkpoint_id] = avg_score
            evaluation_results["details"][checkpoint_id] = {
                "exp_name": eval_exp_name,
                "scores": score_norms,
                "average_score_norm": avg_score
            }

            print(f"✅  Completed evaluation for {checkpoint_id}. Average score_norm: {avg_score:.4f}")

        # ------------------------------------------------------------------
        return evaluation_results

    except Exception as e:
        print(f"Error during optimization evaluation setup: {e}")
        return None

def find_best_checkpoint(evaluation_results):
    """
    Find the best performing checkpoint from evaluation results.
    
    Args:
        evaluation_results: Dictionary containing evaluation statistics
        
    Returns:
        String ID of the best checkpoint
    """
    if not evaluation_results or "win_percentages" not in evaluation_results:
        raise ValueError("Invalid evaluation results format")
        
    # Get the checkpoint with the highest win percentage
    win_percentages = evaluation_results["win_percentages"]
    best_checkpoint_id = max(win_percentages.items(), key=lambda x: x[1])[0]
    
    print(f"\nBest checkpoint identified: {best_checkpoint_id}")
    print(f"Win percentage: {win_percentages[best_checkpoint_id]:.2f}%")
    
    return best_checkpoint_id

def generate_conversations(exp_name, best_checkpoint_id, threshold, num_successful_conversations=110, user_model_id="gpt-4.1-mini-2025-04-14", mode="simple", persona="default"):
    """
    Generate conversations using the selected best model checkpoint as the agent model.
    Continues generating until reaching the target number of successful conversations.
    
    Args:
        exp_name: Name of the experiment
        best_checkpoint_id: ID of the best checkpoint to use as agent model
        num_successful_conversations: Target number of successful conversations to generate
        user_model_id: Model ID to use for the user side
        mode: Either "simple" or "complex" to determine which runner to use
        
    Returns:
        List of paths to output files containing successful conversations
    """
    print(f"\nGenerating {num_successful_conversations} successful conversations using agent model: {best_checkpoint_id}")
    print(f"Conversation mode: {mode}")
    
    # Create a unique base experiment name
    timestamp = int(time.time())
    base_exp_name = f"{exp_name}-run-{timestamp}"
    
    # Keep track of all output files
    all_output_files = []
    
    # Initialize batch size and tracking variables
    batch_size = 200
    total_conversations = 0
    current_successful = 0
    batch_number = 0
    
    # Main output file that will contain the combined successful conversations
    if mode.lower() == "simple":
        main_output_file = f"output_{base_exp_name}_simple.jsonl"
    else:  # complex mode
        main_output_file = f"output_{base_exp_name}.jsonl"
    
    main_output_path = Path(main_output_file)
    
    # Ensure main output file exists even if empty
    if not main_output_path.exists():
        with open(main_output_path, 'w') as f:
            pass
    
    # Use the appropriate parallel_conversations function based on mode
    parallel_conversations = simple_parallel_conversations if mode.lower() == "simple" else complex_parallel_conversations
    
    # Continue generating conversations until we have enough successful ones
    while current_successful < num_successful_conversations:
        # Create a unique experiment name for this batch
        batch_number += 1
        batch_exp_name = f"{base_exp_name}-batch{batch_number}"
        
        # Batch-specific output file
        if mode.lower() == "simple":
            batch_output_file = f"output_{batch_exp_name}_simple.jsonl"
        else:  # complex mode
            batch_output_file = f"output_{batch_exp_name}.jsonl"
        
        batch_output_path = Path(batch_output_file)
        
        # Check how many successful conversations we have so far
        if main_output_path.exists():
            # Count lines in the output file to determine number of successful conversations
            with open(main_output_path, 'r') as f:
                current_successful = sum(1 for _ in f)
            
            print(f"Currently have {current_successful}/{num_successful_conversations} successful conversations")
        else:
            print("No successful conversations yet")
        
        # Calculate how many more conversations we need
        remaining = num_successful_conversations - current_successful
        if remaining <= 0:
            break
            
        # Run another batch of conversations
        next_batch = min(batch_size, 100)  # Generate 2x the remaining needed (assuming ~50% success rate)
        print(f"Generating batch {batch_number} of {next_batch} conversations (attempt to get {remaining} more successful ones)")
        print(f"Using experiment name: {batch_exp_name}")
        
        # Set up model IDs based on mode
        if mode.lower() == "optimization":
            # For optimization mode: best checkpoint as player-1 (user_model_id), gpt-4.1 as player-2 (agent_model_id)
            actual_user_model_id = best_checkpoint_id
            actual_agent_model_id = "gpt-4.1"
            print(f"Optimization mode: player-1={actual_user_model_id}, player-2={actual_agent_model_id}")
        else:
            # For other modes: use provided user_model_id and best checkpoint as agent
            actual_user_model_id = user_model_id
            actual_agent_model_id = best_checkpoint_id
            print(f"Standard mode: user={actual_user_model_id}, agent={actual_agent_model_id}")
        
        # Run conversations in parallel with unique batch name
        parallel_conversations(
            exp_name=batch_exp_name,
            resume=total_conversations,
            end=total_conversations + next_batch,
            num_gpus=batch_size,  # Use available CPUs, max 10
            debug=False,
            timeout_minutes=20,
            track_costs=True,
            user_model_id=actual_user_model_id,
            agent_model_id=actual_agent_model_id,
            threshold=threshold,
            persona=persona
        )
        
        # Update the total number of conversations we've tried
        total_conversations += next_batch
        
        # If batch output file exists, append it to the main output file and our list
        if batch_output_path.exists():
            # Append successful conversations to main output file
            with open(batch_output_path, 'r') as source:
                successful_convs = [line for line in source]
                
            with open(main_output_path, 'a') as destination:
                destination.writelines(successful_convs)
            
            # Add batch file to our list of all output files
            all_output_files.append(str(batch_output_path))
            print(f"Added {len(successful_convs)} conversations from batch {batch_number}")
        else:
            print(f"Warning: Expected batch output file {batch_output_file} not found")
        
        # Avoid hammering the API
        time.sleep(1)
    
    # Add the main output file to our list if not already there
    if str(main_output_path) not in all_output_files:
        all_output_files.append(str(main_output_path))
    
    # Verify final output file
    if not main_output_path.exists():
        raise FileNotFoundError(f"Expected main output file {main_output_file} not found")
    
    # Count final number of successful conversations
    with open(main_output_path, 'r') as f:
        final_count = sum(1 for _ in f)
    
    print(f"Generated {final_count} successful conversations out of {total_conversations} total attempts")
    print(f"Success rate: {final_count/total_conversations*100:.1f}%")
    print(f"Generated conversations saved to: {main_output_file}")
    print(f"All output files: {all_output_files}")
    
    return all_output_files

def format_data_for_finetuning(conversations_files, output_dir, num_conversations, mode="simple"):
    """
    Format the generated conversations for fine-tuning.
    Can handle multiple input files and combines them.
    
    Args:
        conversations_files: List of paths to files containing conversations
        output_dir: Directory to save formatted data
        num_conversations: Maximum number of conversations to include
        mode: Mode to determine which formatting functions to use ("simple", "complex", or "optimization")
        
    Returns:
        Tuple of (agent_file_path, user_file_path)
    """
    if isinstance(conversations_files, str):
        conversations_files = [conversations_files]  # Convert single string to list
        
    print(f"\nFormatting data from {len(conversations_files)} files for fine-tuning")
    for file in conversations_files:
        print(f"  - {file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse conversations from all files
    all_conversations = []
    for file_path in conversations_files:
        # Parse conversations
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    conversation_data = json.loads(line)
                    all_conversations.append(conversation_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}")
    
    print(f"Found {len(all_conversations)} total conversations to format")
    
    # Choose formatting functions based on mode
    if mode.lower() == "optimization":
        agent_formatter = format_for_agent_model_optimization
        user_formatter = format_for_user_model_optimization
        print(f"Using optimization formatting functions for mode: {mode}")
    else:
        agent_formatter = format_for_agent_model
        user_formatter = format_for_user_model
        print(f"Using standard formatting functions for mode: {mode}")
    
    # Format for agent model
    agent_file = os.path.join(output_dir, 'agent_total.jsonl')
    agent_count = agent_formatter(
        all_conversations, 
        agent_file, 
        complex_model=True,  # Based on the format in simple_data.py
        max_length=num_conversations
    )
    
    # Format for user model
    user_file = os.path.join(output_dir, 'user_total.jsonl')
    user_count = user_formatter(
        all_conversations, 
        user_file, 
        complex_model=True,
        max_length=num_conversations
    )
    
    print(f"Formatting complete. Created {agent_count} agent examples and {user_count} user examples")
    return agent_file, user_file

def validate_formatted_data(file_path):
    """
    Validate that the formatted JSONL file follows the expected pattern:
    System message + alternating assistant/user messages.
    
    Args:
        file_path: Path to the formatted JSONL file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    print(f"\nValidating formatted data file: {file_path}")
    
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    error_messages = []
    total_examples = 0
    valid_examples = 0
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                total_examples += 1
                line_errors = []
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    line_errors.append(f"Invalid JSON: {e}")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue
                
                # Check if it has the expected structure
                if "messages" not in data:
                    line_errors.append("Missing 'messages' field")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    line_errors.append("'messages' should be a list with at least 2 messages")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue
                
                # Validate message structure and pattern
                expected_roles = []
                
                # First message should be system
                if messages[0].get("role") != "system":
                    line_errors.append("First message should have role 'system'")
                else:
                    expected_roles.append("system")
                    
                    # Determine the alternating pattern for the rest
                    # After system, it should alternate between assistant and user
                    # We need to check what the second message is to determine the pattern
                    if len(messages) > 1:
                        second_role = messages[1].get("role")
                        if second_role in ["assistant", "user"]:
                            current_role = second_role
                            expected_roles.append(current_role)
                            
                            # Alternate for the rest
                            for i in range(2, len(messages)):
                                if current_role == "assistant":
                                    current_role = "user"
                                else:
                                    current_role = "assistant"
                                expected_roles.append(current_role)
                        else:
                            line_errors.append(f"Second message should be 'assistant' or 'user', got '{second_role}'")
                
                # Check each message
                for i, (message, expected_role) in enumerate(zip(messages, expected_roles)):
                    if not isinstance(message, dict):
                        line_errors.append(f"Message {i+1} should be a dictionary")
                        continue
                        
                    if "role" not in message:
                        line_errors.append(f"Message {i+1} missing 'role' field")
                        continue
                        
                    if "content" not in message:
                        line_errors.append(f"Message {i+1} missing 'content' field")
                        continue
                        
                    actual_role = message.get("role")
                    if actual_role != expected_role:
                        line_errors.append(f"Message {i+1} expected role '{expected_role}', got '{actual_role}'")
                        
                    if not message.get("content", "").strip():
                        line_errors.append(f"Message {i+1} has empty content")
                
                # If no errors for this line, it's valid
                if not line_errors:
                    valid_examples += 1
                else:
                    for error in line_errors:
                        error_messages.append(f"Line {line_num}: {error}")
                        
    except Exception as e:
        error_messages.append(f"Error reading file: {e}")
        return False, error_messages
    
    # Print validation summary
    print(f"Validation Results:")
    print(f"  Total examples: {total_examples}")
    print(f"  Valid examples: {valid_examples}")
    print(f"  Invalid examples: {total_examples - valid_examples}")
    print(f"  Success rate: {valid_examples/total_examples*100:.1f}%" if total_examples > 0 else "  Success rate: N/A")
    
    if error_messages:
        print(f"  Found {len(error_messages)} validation errors:")
        # Show first 10 errors to avoid overwhelming output
        for error in error_messages[:10]:
            print(f"    {error}")
        if len(error_messages) > 10:
            print(f"    ... and {len(error_messages) - 10} more errors")
    else:
        print("  All examples are valid!")
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages

def finetune_checkpoint(client, training_file, best_checkpoint_id, suffix, train_user=False):
    """
    Fine-tune the best checkpoint using the formatted data.
    
    Args:
        client: OpenAI client
        training_file: Path to formatted training data file (agent or user)
        best_checkpoint_id: ID of the best checkpoint to fine-tune
        suffix: Suffix for the new fine-tuned model
        train_user: If True, train user model; if False, train agent model
        
    Returns:
        Job ID of the fine-tuning job
    """
    model_type = "user" if train_user else "agent"
    print(f"\nStarting fine-tuning of {model_type} model using checkpoint: {best_checkpoint_id}")
    
    # Create output directory
    output_dir = os.path.join("finetune", suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command to run finetune_parallel.py
    # Get the absolute path to finetune_parallel.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finetune_parallel_path = os.path.join(script_dir, "finetune_parallel.py")
    cmd = [sys.executable, finetune_parallel_path]
    
    # Add required arguments
    cmd.extend(["--suffix", suffix])
    cmd.extend(["--model", best_checkpoint_id])
    
    # Use appropriate training file and flag based on train_user parameter
    if train_user:
        cmd.extend(["--user_jsonl", training_file])
        cmd.extend(["--train_user"])  # Train the user model
        print(f"Training user model with data from: {training_file}")
    else:
        cmd.extend(["--agent_jsonl", training_file])
        cmd.extend(["--train_agent"])  # Train the agent model (default)
        print(f"Training agent model with data from: {training_file}")
    
    cmd.extend(["--output_dir", output_dir])
    #cmd.extend(["--n_epochs", "3"])
    #cmd.extend(["--batch_size", "2"])
    #cmd.extend(["--learning_rate_multiplier", "8.0"])
    
    # Run the command
    print(f"Running fine-tuning with command:")
    print(" ".join(cmd))
    
    # Execute the command and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    job_id = None
    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print(output_line.strip())
            # Try to capture the job ID from output
            if "job ID:" in output_line:
                job_id_match = output_line.strip().split("job ID:")[1].strip()
                if job_id_match:
                    job_id = job_id_match
    
    return_code = process.poll()
    
    if return_code == 0:
        print(f"Fine-tuning job completed successfully")
        return job_id
    else:
        print(f"Fine-tuning job failed with return code {return_code}")
        return None

def run_whole_pipeline(args):
    """
    Run the complete pipeline from evaluation to fine-tuning.
    
    Args:
        args: Command-line arguments
    """
    client = OpenAI()
    
    print("====== STARTING FINE-TUNING PIPELINE ======")
    
    # Variables to store intermediate outputs
    evaluation_results = None
    best_checkpoint_id = None
    conversations_files = None
    agent_file = None
    user_file = None
    job_id = None
    
    # Determine starting step
    start_step = args.start_step
    current_step = 1
    
    # Step 1: Evaluate checkpoints
    if current_step >= start_step:
        print("\n----- STEP 1: Evaluating Checkpoints -----")
        
        # Comment out previous implementation
        # evaluation_results = evaluate_latest_job(
        #     client=client,
        #     validation_file=args.validation_file,
        #     job_id=args.job_id,
        #     output_file=args.output_file,
        #     judge_model=args.judge_model
        # )
        
        # Create argument namespace for evaluation
        from argparse import Namespace
        eval_args = Namespace(
            exp_name=args.exp_name,
            user_model_id=args.user_model_id,
            job_id=args.job_id,
            mode=args.mode,
            num_conversations=10,
            output_file=None,
            parallel_evals=None,
            persona=args.persona,  # Add persona parameter
            eval_side=args.eval_side  # New: which player to evaluate
        )
        
        # Use appropriate evaluation function based on mode and parallel setting
        if args.mode == "optimization":
            if args.parallel:
                print("Using parallel optimization evaluation...")
                evaluation_results = evaluate_optimization_checkpoints_parallel(eval_args)
            else:
                print("Using sequential optimization evaluation...")
                evaluation_results = evaluate_optimization_checkpoints(eval_args)
        else:
            print("Using standard evaluation...")
            evaluation_results = evaluate_optimization_checkpoints(eval_args)
        
        if not evaluation_results:
            print("Error: No evaluation results returned. Exiting pipeline.")
            return
        
        print("Step 1 complete. Entering debugger to verify evaluation results...")
        pdb.set_trace()  # Breakpoint after step 1
    
    # Step 2: Identify best checkpoint
    current_step += 1
    if current_step >= start_step:
        print("\n----- STEP 2: Identifying Best Checkpoint -----")
        
        # If skipping the previous step, use the provided best checkpoint
        if evaluation_results:
            best_checkpoint_id = find_best_checkpoint(evaluation_results)
        elif args.best_checkpoint:
            best_checkpoint_id = args.best_checkpoint
            print(f"Using provided best checkpoint: {best_checkpoint_id}")
        else:
            print("Error: Best checkpoint not provided when skipping step 1.")
            return
        
        print("Step 2 complete. Entering debugger to verify best checkpoint...")
        pdb.set_trace()  # Breakpoint after step 2
    
    # Step 3: Generate conversations
    current_step += 1
    if current_step >= start_step:
        print("\n----- STEP 3: Generating Conversations -----")
        
        # If skipping previous steps, use the provided best checkpoint
        checkpoint_to_use = best_checkpoint_id or args.best_checkpoint
        
        if not checkpoint_to_use:
            print("Error: Best checkpoint not available for conversation generation.")
            return
            
        if args.conversations_file and start_step > current_step:
            # Use provided conversations file if skipping this step
            conversations_files = args.conversations_file
            print(f"Using provided conversations files: {conversations_files}")
        else:
            # Generate new conversations
            conversations_files = generate_conversations(
                exp_name=args.exp_name,
                best_checkpoint_id=checkpoint_to_use,
                num_successful_conversations=args.num_conversations,
                user_model_id=args.user_model_id,
                threshold = args.threshold,
                mode=args.mode,
                persona=args.persona
            )
            conversations_files = conversations_files[:-1]
        
        print("Step 3 complete. Entering debugger to verify conversations files...")
        pdb.set_trace()  # Breakpoint after step 3
    
    # Step 4: Format data for fine-tuning
    current_step += 1
    if current_step >= start_step:
        print("\n----- STEP 4: Formatting Data -----")
        
        # If skipping previous step, use the provided conversations file
        conv_files_to_use = conversations_files or args.conversations_file
        
        if not conv_files_to_use or not all(conv_files_to_use):
            print("Error: Conversations files not available for formatting.")
            return
            
        if args.agent_file and start_step > current_step:
            # Use provided agent file if skipping this step
            agent_file = args.agent_file
            # For user file, try to infer the path or use the same directory
            if args.agent_file:
                agent_dir = os.path.dirname(args.agent_file)
                user_file = os.path.join(agent_dir, 'user_total.jsonl')
                if not os.path.exists(user_file):
                    user_file = None
            print(f"Using provided agent file: {agent_file}")
            if user_file:
                print(f"Using inferred user file: {user_file}")
        else:
            # Format the data from all conversation files
            output_dir = os.path.join("finetune", f"{args.exp_name}_iteration")
            agent_file, user_file = format_data_for_finetuning(conv_files_to_use, output_dir, args.num_conversations, args.mode)
        
        # Determine which file to validate based on ft_user flag
        training_file_to_validate = user_file if args.ft_user else agent_file
        model_type = "user" if args.ft_user else "agent"
        
        if not training_file_to_validate:
            print(f"Error: {model_type} training file not available.")
            return
        
        # Validate the formatted data
        print(f"\n----- STEP 4a: Validating Formatted {model_type.title()} Data -----")
        is_valid, validation_errors = validate_formatted_data(training_file_to_validate)
        
        if not is_valid:
            print(f"Warning: Validation found {len(validation_errors)} errors in the formatted {model_type} data.")
            print("The pipeline will continue, but you may want to review the formatting function.")
        else:
            print(f"✓ Formatted {model_type} data validation passed!")
        
        print("Step 4 complete. Entering debugger to verify formatted data...")
        pdb.set_trace()  # Breakpoint after step 4
    
    # Step 5: Fine-tune the best checkpoint
    current_step += 1
    if current_step >= start_step:
        print("\n----- STEP 5: Fine-tuning Best Checkpoint -----")
        
        # If skipping previous steps, use the provided parameters
        checkpoint_to_use = best_checkpoint_id or args.best_checkpoint
        
        # Determine which training file to use based on ft_user flag
        if args.ft_user:
            training_file_to_use = user_file or (args.agent_file and os.path.join(os.path.dirname(args.agent_file), 'user_total.jsonl'))
            model_type = "user"
        else:
            training_file_to_use = agent_file or args.agent_file
            model_type = "agent"
        
        if not checkpoint_to_use:
            print("Error: Best checkpoint not available for fine-tuning.")
            return
            
        if not training_file_to_use or not os.path.exists(training_file_to_use):
            print(f"Error: {model_type.title()} training file not available for fine-tuning.")
            return
            
        print(f"Fine-tuning {model_type} model using: {training_file_to_use}")
        
        # Fine-tune the checkpoint
        job_id = finetune_checkpoint(
            client=client,
            training_file=training_file_to_use,
            best_checkpoint_id=checkpoint_to_use,
            suffix=f"{args.exp_name}_iter",
            train_user=args.ft_user
        )
        
        print("Step 5 complete. Entering debugger to verify fine-tuning job...")
        pdb.set_trace()  # Breakpoint after step 5
    
    print("\n====== PIPELINE COMPLETED SUCCESSFULLY ======")
    if evaluation_results:
        print(f"Evaluation results saved to: {args.output_file}")
    if best_checkpoint_id:
        print(f"Best checkpoint: {best_checkpoint_id}")
    if conversations_files:
        print(f"Generated conversation files: {conversations_files}")
    if agent_file:
        print(f"Formatted agent training data: {agent_file}")
    if user_file:
        print(f"Formatted user training data: {user_file}")
    if job_id:
        model_type = "user" if args.ft_user else "agent"
        print(f"Fine-tuning job ID ({model_type} model): {job_id}")
    
    return {
        "best_checkpoint": best_checkpoint_id or args.best_checkpoint,
        "conversations_files": conversations_files or args.conversations_file,
        "agent_training_data": agent_file or args.agent_file,
        "user_training_data": user_file,
        "fine_tuning_job": job_id,
        "trained_model_type": "user" if args.ft_user else "agent"
    }

def main():
    parser = argparse.ArgumentParser(description='Run complete fine-tuning pipeline')
    # Required parameters
    parser.add_argument('--exp_name', required=True, help='Name of the experiment')
    
    # Basic parameters
    parser.add_argument('--validation_file', help='Path to validation data file (required for step 1)')
    parser.add_argument('--job_id', help='Specific job ID (if not provided, uses latest job)')
    parser.add_argument('--output_file', help='Path to output file for evaluation results')
    parser.add_argument('--judge_model', default='gpt-4.1-mini-2025-04-14', help='Model to use as judge')
    parser.add_argument('--num_conversations', type=int, default=110, help='Number of conversations to generate')
    parser.add_argument('--user_model_id', default='gpt-4.1', help='Model to use for user side in conversation generation')
    parser.add_argument('--mode', required=True, choices=['simple', 'complex', 'optimization'], help='Conversation mode (simple, complex, or optimization)')
    parser.add_argument('--threshold', type=float, help='Threshold for successful conversations (required for complex and optimization modes)')
    # Step selection and intermediate data
    parser.add_argument('--start_step', type=int, default=1, choices=range(1, 6), 
                        help='Step to start from (1:Evaluate, 2:FindBest, 3:Generate, 4:Format, 5:Finetune)')
    parser.add_argument('--best_checkpoint', help='Best checkpoint ID (required if skipping step 1-2)')
    parser.add_argument('--conversations_file', nargs='*',
                    type=str,help='Path to existing conversations file(s) before formating; comma or space separated (required if starting from step 4)')
    parser.add_argument('--agent_file', help='Path to formatted training data file - can be agent or user data (required if starting from step 5)')
    parser.add_argument('--persona', required=True, choices=['default', 'shy'], help='Persona to use for conversation generation')
    parser.add_argument('--parallel', action='store_true', help='Use parallel execution for checkpoint evaluation (faster, default for step 1)')
    parser.add_argument('--eval_side', choices=['player1', 'player2'], default='player2',
                        help='Which player the checkpoint model should take during evaluation in step 1 (player1 or player2).')
    parser.add_argument('--ft_user', action='store_true', help='Fine-tune the user model instead of the agent model (default: fine-tune agent)')
    args = parser.parse_args()
    

# After parsing

    

    def parse_file_list(value):
        # If it's a single string that might contain multiple files
        value = ' '.join(value) if isinstance(value, list) else value
        
        # Remove outer brackets if present
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        
        # Try to parse as Python literal first
        try:
            parsed = ast.literal_eval(f"[{value}]")
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        
        # Split by comma or space
        if ',' in value:
            files = [item.strip().strip("'\"") for item in value.split(',')]
        else:
            files = [item.strip().strip("'\"") for item in value.split()]
        
        return [f for f in files if f]



    # After parsing
  
    if args.conversations_file:
        args.conversations_file = parse_file_list(args.conversations_file)
    
    # Validate arguments based on starting step
    if args.start_step > 1 and not args.validation_file and not args.best_checkpoint:
        parser.error("--best_checkpoint is required when skipping evaluation steps")
    
    if args.start_step == 4 and not args.conversations_file:
        parser.error("--conversations_file is required when starting from formatting step")
    
    if args.start_step == 5 and not args.agent_file:
        parser.error("--agent_file is required when starting from fine-tuning step")
    
    # Validate threshold based on mode
    if args.mode in ['complex', 'optimization'] and args.threshold is None:
        parser.error("--threshold is required when mode is 'complex' or 'optimization'")
    
    # Print information about fine-tuning mode
    if args.ft_user:
        print("Pipeline configured to fine-tune USER model")
    else:
        print("Pipeline configured to fine-tune AGENT model (default)")
    
    # Run the pipeline
    run_whole_pipeline(args)

if __name__ == "__main__":
    main()
