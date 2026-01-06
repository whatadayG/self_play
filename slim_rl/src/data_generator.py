"""Generate game instances in SLIME's expected JSONL format.

This module creates paper-reviewer matching game instances that can be
used for training with SLIME. Each game instance includes:
- A prompt with the agent's visible scores
- The full game state for environment reset
- Metadata like optimal score and game ID
"""
import json
import sys
from pathlib import Path
from typing import Optional

# Add self_play/scripts to path for dialop imports
SCRIPTS_PATH = Path(__file__).parent.parent.parent / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from dialop.games.optimization import OptimizationGame, TASKS, WORKERS


# System prompt for the agent (Qwen)
AGENT_SYSTEM_PROMPT = """You are playing a paper-reviewer matching game. Your goal is to assign 8 reviewers to 8 papers to maximize the total affinity score.

GAME RULES:
- You can see SOME of the affinity scores between reviewers and papers
- Your partner can see OTHER scores that you cannot see (shown as empty cells in your table)
- Scores you don't know are marked with "-" in your table
- You must collaborate with your partner to find the best assignment

TO WIN, you should:
1. ASK your partner about specific scores you don't know (e.g., "What is the score for Ava Li reviewing BLEU?")
2. SHARE information when your partner asks about scores you can see
3. PROPOSE assignments when you think you have enough information
4. ACCEPT or REJECT your partner's proposals based on your knowledge

FORMAT FOR PROPOSALS:
[propose]
- Ava Li: BLEU
- Daniel Nguyen: Electra
- Sofia Patel: GloVe
(... assign all 8 reviewers to 8 papers)
[/propose]

YOUR VISIBLE AFFINITY SCORES:
{agent_view}

Remember: Your partner has information you don't have. Ask questions to gather the information you need!"""


def format_table_view(table: list, include_headers: bool = True) -> str:
    """Format a table view as a readable string.

    Args:
        table: 2D list with headers (first row = papers, first col = reviewers)
        include_headers: Whether to include row/column headers

    Returns:
        Formatted table string
    """
    if not table:
        return ""

    # Table already includes headers from _preprocess_table
    lines = []
    for row in table:
        # Convert each cell to string, handling empty strings
        cells = []
        for cell in row:
            if cell == "" or cell is None:
                cells.append("-")
            else:
                cells.append(str(cell))
        lines.append(" | ".join(f"{c:>12}" for c in cells))

    return "\n".join(lines)


def generate_game(seed: int, filter_hard: bool = True) -> dict:
    """Generate a single game instance.

    Args:
        seed: Random seed for deterministic generation
        filter_hard: If True, only generate "hard" games where collaboration helps

    Returns:
        dict with: prompt, game_state, ground_truth, optimal_score, game_id
    """
    game = OptimizationGame({}, one_player=False)
    game.reset(seed=seed)

    # Get game info for serialization
    game_info = game.get_game_info()

    # Get agent's (player 0) view as formatted table
    # tables[0] is already preprocessed with headers
    agent_table = game.tables[0]
    agent_view = format_table_view(agent_table)

    # Build the prompt with agent's view
    prompt = AGENT_SYSTEM_PROMPT.format(agent_view=agent_view)

    # Build game state that can be used to reset the environment
    game_state = {
        "table": game_info["table"],
        "mask1": game_info["mask1"],
        "mask2": game_info["mask2"],
        "scale1": game_info["scale1"],
        "scale2": game_info["scale2"],
        "best_assignment_reward": game_info["best_assignment_reward"],
        "action_log": [],
    }

    return {
        "prompt": prompt,
        "game_state": game_state,
        "ground_truth": game_state,  # SLIME sometimes uses this key
        "optimal_score": float(game.best_assignment_reward),
        "game_id": seed,
    }


def generate_dataset(
    num_games: int,
    output_path: str,
    base_seed: int = 42,
    filter_hard: bool = True,
    verbose: bool = True,
) -> None:
    """Generate a dataset of game instances.

    Args:
        num_games: Number of games to generate
        output_path: Path to write JSONL file
        base_seed: Base random seed
        filter_hard: If True, only include "hard" games
        verbose: If True, print progress
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    seed = base_seed

    with open(output_path, "w") as f:
        while generated < num_games:
            try:
                game = generate_game(seed, filter_hard=filter_hard)
                f.write(json.dumps(game) + "\n")
                generated += 1

                if verbose and generated % 100 == 0:
                    print(f"Generated {generated}/{num_games} games...")

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to generate game with seed {seed}: {e}")

            seed += 1

            # Safety check to avoid infinite loop
            if seed - base_seed > num_games * 10:
                print(f"Warning: Too many failed generations, stopping at {generated} games")
                break

    if verbose:
        print(f"Generated {generated} games to {output_path}")


def load_dataset(path: str) -> list[dict]:
    """Load a dataset from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of game instances
    """
    games = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                games.append(json.loads(line))
    return games


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper-reviewer matching games")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter for hard games")

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = str(Path(__file__).parent.parent / "data" / "games.jsonl")

    generate_dataset(
        num_games=args.num_games,
        output_path=args.output,
        base_seed=args.seed,
        filter_hard=not args.no_filter,
    )
