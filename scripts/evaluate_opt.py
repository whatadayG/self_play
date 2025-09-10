from copy import deepcopy
import json
import os
import pathlib
import time
from typing import Optional, Literal
from itertools import cycle
import pdb
import random
import pickle
from pathlib import Path


from openai import APIConnectionError, RateLimitError
import numpy as np
import tyro
from ruamel.yaml import YAML
from rich import print
from rich.console import Console
from rich.markup import escape
console = Console()

from dialop.envs import (
    OptimizationEnv,
    WordLimit,
    ForceProposal,
    AsymmetricForceProposal
)
from dialop.players import (
    LLMPlayer,
    HumanPlayer,
    DryRunPlayer,
    OutOfContextError
)
try:
    from dialop.local_model_player import LocalModelPlayer, LocalModelPlayerVLLM
except ImportError:
    LocalModelPlayer = None
    LocalModelPlayerVLLM = None

try:
    from dialop.hf_model_player import HFModelPlayer
except ImportError:
    HFModelPlayer = None
from dialop.utils import Logger, retry, count_words
from dialop.simple.simple_api_tracker import OpenAITracker
from dialop.simple.cost_tracker import CostTracker
from dialop.sglang_model_player import SglangModelPlayer

PROJECT_ROOT = Path(__file__).parent

FPATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
RESDIR = PROJECT_ROOT / "results"
DATADIR = PROJECT_ROOT / "data"

GAME_CLSS = {
    "matching": OptimizationEnv,
}
def save_itinerary_data(game_cls, conversations, player1_agent, player2_user, id, reward_normalized=None, output_path="output_itinerary.jsonl", env=None):
    """
    Save conversation and itinerary data to a JSONL file similar to itinerary.jsonl.
    
    Args:
        conversations: List of conversation messages
        player1_agent: Player 1 data
        player2_user: Player 2 data
        id: Game ID
        reward_normalized: Normalized reward (0-1)
        output_path: Path to save the JSONL file
        env: Game environment to extract full game state
    """
    
    # Process events to ensure they're serializable
    if isinstance(player1_agent, list) and len(player1_agent) > 0 and isinstance(player1_agent[0], dict):
        processed_player1_agent = player1_agent
    else:
        # Handle other cases (like domain knowledge string)
        processed_player1_agent = player1_agent
    
    if game_cls == OptimizationEnv:
        # Get the full game info including action_log
        game_info = env.game.get_game_info() if env else {}
        
        entry = {
            "conversations": conversations,
            "player_1_data": processed_player1_agent,
            "player_2_data": player2_user,
            "id": id,
            "reward_normalized": reward_normalized,
            "game_info": game_info,  # Full game state with action_log
            "action_log": game_info.get("action_log", []),  # Explicit action log for easy access
            "proposal_reward": game_info.get("proposal_reward", 0),
            "best_assignment_reward": game_info.get("best_assignment_reward", 0),
            "random_seed": random.randint(1, 10000)  # Add random seed for debugging
        }
    else:
        # Keep backward compatibility for other game types
        entry = {
            "conversations": conversations,
            "player_1_data": processed_player1_agent,
            "player_2_data": player2_user,
            "id": id,
            "reward_normalized": reward_normalized,
            "random_seed": random.randint(1, 10000)
        }
    
    # Write to JSONL file
    with open(f"{output_path}", 'a') as f:
        f.write(json.dumps(entry) + '\n')
    
    print(f"Itinerary data saved to {output_path}")
class ResampleError(Exception):
    pass

def selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end,
):
    """Generator that yields game states for evaluation.

    If *games* is non-empty we iterate over the provided objects.  These
    objects can be in one of two formats:

    1. The full historic game dictionary produced by previous runs which
       contains an ``action_log`` and ``result`` information.
    2. A simplified wrapper of the form ``{"id": ..., "game_cls": ...,
       "game_state": {...}}`` coming from an external JSONL supplied via
       the new ``--load-game-state`` flag.  In this case we lift the inner
       ``game_state`` dict and treat missing metadata as empty / zero.
    """

    if games:
        for game_idx, game in enumerate(games[resume:end]):
            # Detect wrapper format produced by the new --load-game-state flag
            if "game_state" in game:
                data = deepcopy(game["game_state"])
                original_log = data.get("action_log", [])

                # External files may not contain score metadata; default to 0.
                score = game.get("score", 0)
                score_norm = game.get("score_norm", 0)
            else:
                original_log = game.get("action_log", [])
                data = deepcopy(game)
                # Clear action log so env doesn't initialize with a message history
                data["action_log"] = []

                # Extract scores if present
                if game_cls == OptimizationEnv:
                    score = data.get("proposal_reward", 0)
                    score_norm = data.get("result", {}).get("norm", 0)
                else:
                    score = data.get("result", {}).get("score", 0)
                    score_norm = data.get("result", {}).get("norm", 0)

            metadata = {
                "hh_turns": len(original_log),
                "hh_words": count_words(original_log),
                "hh_score": score,
                "hh_score_norm": score_norm,
            }

            for sidx in range(samples_per_game):
                name = f"{game_idx + resume}_{sidx}"
                yield data, name, metadata
    else:
        # No pre-existing games: fall back to random generation.
        for i in range(resume, end):
            name = f"{i}"
            yield None, name, None

def full_conversation_reproposal(
    game_cls,
    games, 
    samples_per_game,
    resume,
    end,
):
    """Generator that yields game states with 100% conversation history but last proposal removed.
    
    This function:
    1. Uses 100% of the game state and chat history 
    2. Removes the last proposal and any subsequent responses
    3. Automatically detects who made the original final proposal 
    4. Sets that same player to make the new proposal
    5. The non-proposer automatically responds with "Lets think step by step. [accept]"
    6. Both players remain as LLMPlayer objects but auto-acceptance is handled in the game loop
    """
    for game_idx, game in enumerate(games[resume:end]):
        if game_cls == OptimizationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
            
            # Find the last proposal in the action log and who made it
            last_proposal_idx = None
            original_proposer_idx = None
            for turn in range(len(original_log) - 1, -1, -1):  # Search backwards
                if original_log[turn].get("type") == "proposal":
                    last_proposal_idx = turn
                    original_proposer_idx = original_log[turn]["player"]
                    break
            
            if last_proposal_idx is None:
                raise ValueError("Game doesn't include any proposal")
            
            # Convert player index to player name
            original_proposer = f"player-{original_proposer_idx + 1}"
            
            # Remove the last proposal and everything after it
            truncated_log = original_log[:last_proposal_idx]
            data["action_log"] = truncated_log
            
            # Calculate scores from the original final proposal
            try:
                score = data["proposal_reward"]
            except:
                # Find the proposal reward from the original log
                for turn in range(0, len(original_log)):
                    if original_log[turn]["type"] == "proposal":
                        score = original_log[turn].get("scores", {}).get("total", 0)
                        
        # matching-only
        metadata = {
            "initialized_turns": len(data["action_log"]),
            "initialized_words": count_words(data["action_log"]),
            "hh_turns": len(original_log),
            "hh_words": count_words(original_log),
            "hh_score": score,
            "hh_score_norm": data.get("result", {}).get("norm", 0),
            "mode": "full_conversation_reproposal",
            "original_proposer": original_proposer  # Track who originally proposed
        }
        
        for sidx in range(samples_per_game):
            name = f"{game_idx + resume}_full_reproposal_{original_proposer}_{sidx}"
            yield data, name, metadata

def prompted_selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end,
    mode="standard"  # Add mode parameter
):
    for game_idx, game in enumerate(games[resume:end]):
        if game_cls == OptimizationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        elif game_cls == OptimizationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        # matching-only

        if game_cls == OptimizationEnv:
            try:
                score = data["proposal_reward"]
            except:
                for turn in range(0, len(data["action_log"])):
                    if data["action_log"][turn]["type"] == "proposal":
                        score = data["action_log"][turn]["scores"]["total"]
        # matching-only

        total_word_count = count_words(original_log)
        prefix_word_counts = []
        for turn in range(0, len(data["action_log"])):
            num_words = count_words(original_log[:turn])
            prefix_word_counts.append(num_words)
        # Get turns closest to 25%, 50%, 75% of the way through the game:
        turn_idxs = []
        # for pct in [0.25, 0.5, 0.75]:
        for pct in [0.5, 0.75]:
            turn_idxs.append(
                np.argmin(np.abs(np.array(prefix_word_counts) - pct * total_word_count))
            )
        # Get index of final proposal:
        proposal_idx = None
        for turn in range(0, len(data["action_log"])):
            if data["action_log"][turn]["type"] == "proposal":
                proposal_idx = turn
        if proposal_idx is None:
            raise ValueError("Game doesn't include a proposal")
        turn_idxs.append(proposal_idx)

        # import pdb; pdb.set_trace()
        # for turn in range(0, len(data["action_log"])):
        names = ["50", "75", "end"]
        # for turn in range(2, len(data["action_log"]) // 2):
        #     end = 2 * (turn + 1)
        for name, end in zip(names, turn_idxs):
            # if end >= len(game["games"][0]["action_log"]): continue
            data["action_log"] = original_log[:end]
            metadata = {
                "initialized_turns": len(data["action_log"]),
                "initialized_words": count_words(data["action_log"]),
                "hh_turns": len(original_log),
                "hh_words": count_words(original_log),
                "hh_score": score,
                "hh_score_norm": data["result"]["norm"],
            }
            for sidx in range(samples_per_game):
                name = f"{game_idx + resume}_start{len(data['action_log'])}_{name}_{sidx}"
                yield data, name, metadata

#@retry(allowed_exceptions=[OutOfContextError, RateLimitError, ResampleError])
def run(
    game_cls,
    data,
    metadata,
    player_ctor,
    env_ctor,
    logfile,
    use_word_limit=False,
    max_length=35,
    exp_name = None,
    threshold = 0.5
): # currently threshol is only for planning
    # Create players and environment
    players = player_ctor()
    env = env_ctor()
    
    # Reset environment with game state
    if use_word_limit:
        obss, domain_knowledge = env.reset(word_limit=metadata["hh_words"],
                         game_state=data)
    else:
        if game_cls == OptimizationEnv:
            obss = env.reset(game_state=data)
        else:
            obss, domain_knowledge = env.reset(game_state=data)
    
    # Update prompts with scale-specific information for two-player optimization mode
    if game_cls == OptimizationEnv:
        # Calculate unknown value as 50 * scale for each player
        for pname, player in players.items():
            player_idx = 0 if pname == "player-1" else 1
            scale = env.game.scales[player_idx]
            unknown_value = int(50 * scale)
            # Update the prompt by replacing the placeholder
            if "{unknown_value}" in player.prompt:
                player.prompt = player.prompt.replace("{unknown_value}", str(unknown_value))

    # CRITICAL: Explicit turn_player control for full_conversation_reproposal mode
    if metadata and metadata.get("mode") == "full_conversation_reproposal":
        proposer = metadata.get("original_proposer", "player-1") # Use original_proposer from metadata
        if game_cls == OptimizationEnv:
            # Set turn_player explicitly to control who proposes
            # player-1 = index 0, player-2 = index 1
            env.game.turn_player = 0 if proposer == "player-1" else 1
            
            # Update observations to reflect the correct turn player
            obss["turn_player"] = env.players[env.game.turn_player]
            
            print(f"Explicit turn control: {proposer} (index {env.game.turn_player}) will make the proposal")

    # Log initial info
    log = Logger(logfile)
  
    # Set up API trackers for players
    for pname, player in players.items():
        api_log_file = str(logfile).replace('.out', f'_{pname}_api.json')
        player.api_tracker = OpenAITracker(api_log_file)
    


    for pname, player in players.items():
        log.write(
            f"{pname} params",
            json.dumps(getattr(player, 'model_kwargs', {}), indent=2))
        log.write(f"{pname} prompt", player.prompt)
    
    # Env loop
    t = 0
    
    # Track if we need to auto-respond to a proposal in full_conversation_reproposal mode
    auto_respond_to_proposal = False
    

    #import pdb; pdb.set_trace() !!!! !domain_knowlege is wrong- the same always somehow 
    # Matching-only env loop
    if True:
        player_1_data = obss["player-1"]
        player_2_data = obss["player-2"]
        assert len(players['player-1'].model_format) == 1 and len(players['player-2'].model_format) == 1, "players model format at the begining must be 1 (system)"
        
        players['player-1'].model_format[0]['content'] += player_1_data
        players['player-2'].model_format[0]['content'] += player_2_data
        conversations_save = []
 
        while not obss["done"] and t < max_length:
            console.rule("environment obs")
            console.print(obss)
            
            # Have players observe the current state
            ##if obss["player-1"].startswith("Reviewer Paper Similarity Scores:") or obss["player-2"].startswith("Reviewer Paper Similarity Scores:"):
            ##    info = True
            ##else:
            ##    info = False
      
            [player.observe(obss[pname]) for pname, player in players.items()]
            
            # Log observations

            for pname in players:
                log.log(key=pname, value=obss[pname], title=f"obs t={t}")
            
            # Get current player
            current_player = obss["turn_player"]
            
            # Handle response and resampling
            
         
            
            # Force proposal near the end of conversation
            if game_cls == OptimizationEnv:
                force_proposal = t > (max_length - 5)
            else:
                force_proposal = False
            

                    # Let the player observe the error
                
                # Get response from current player
            resample_count = 0
            refresh_count = 0

            if game_cls == OptimizationEnv:
                player_1_base_prompt = players['player-1'].prompt
                player_2_base_prompt = players["player-2"].prompt
            else:
                agent_base_prompt = None
                user_base_prompt = None
            while True:
                if refresh_count >= 20:
                    # Skip this run and move to the next fname
                    print(f"Exceeded maximum refresh attempts ({refresh_count}). Skipping to next run.")
                    log.write("Exceeded maximum refresh attempts", f"Exceeded maximum refresh attempts ({refresh_count}). Skipping to next run.")
                    log.flush()
                    log.close()
                    return  # Exit the current run function
                if resample_count >= 5:
                    if game_cls == OptimizationEnv:
                        if current_player == "player-1":
                            players['player-1'].prompt = player_1_base_prompt
                        else:
                            players["player-2"].prompt = player_2_base_prompt
                try:
                    #import pdb; pdb.set_trace()
                    # Decide whether to force a proposal for **this** turn.
                    # We only force when (a) we are in the final window AND
                    # (b) there is **no** active proposal yet.  Once a
                    # proposal exists, the other player must send [accept] or
                    # [reject] without the propose flag.
                    must_propose_now = (force_proposal and getattr(env.game, "proposal", None) is None) or (metadata and metadata.get("mode") == "full_conversation_reproposal" and not auto_respond_to_proposal)

                    # SPECIAL LOGIC: Auto-respond to proposals in full_conversation_reproposal mode
                    if (metadata and metadata.get("mode") == "full_conversation_reproposal" and 
                        auto_respond_to_proposal):
                        # Automatically respond with acceptance message instead of calling LLM
                        resp = "Lets think step by step. [accept]"
                        auto_respond_to_proposal = False  # Reset flag
                        print(f"Auto-responding to proposal: {resp}")
                    elif must_propose_now:
                        if hasattr(players[current_player], 'model_kwargs'):  # LLMPlayer
                            resp = players[current_player].respond(t=t, propose=True)
                        else:  # DryRunPlayer
                            players[current_player].prompt += "System: Remember, you must include '[message]' or '[propose]' or '[accept]' or '[reject]' tags in your response."
                            resp = players[current_player].respond()
                    else:
                        if hasattr(players[current_player], 'model_kwargs'):  # LLMPlayer
                            resp = players[current_player].respond(t=t)
                        else:  # DryRunPlayer
                            resp = players[current_player].respond()
                except RateLimitError:
                    resp = ""
                    print("Rate limited. Sleeping...")
                    time.sleep(10)
                
                # Apply the same logic when we actually hand the response to
                # the environment: only set the `propose` flag when we really
                # forced the agent to propose.
                players[current_player].model_format.append({"role": "assistant", "content": resp})
                
                if must_propose_now:
                    obss, resample = env.step(resp, propose=True, user_think=True)
                else:
                    obss, resample = env.step(resp, user_think=True)
                    print(f"obss: {obss}")
                    
                               #import pdb; pdb.set_trace()
                
                # DETECTION: Check if a proposal was just made in full_conversation_reproposal mode
                if (metadata and metadata.get("mode") == "full_conversation_reproposal" and 
                    not resample and "[propose]" in resp.lower()):
                    # A proposal was just made, set flag to auto-respond on next turn
                    auto_respond_to_proposal = True
                    print("Proposal detected - will auto-respond on next turn")
                
                if not resample:
                    if 'Error' in obss['player-1'] or 'Error' in obss['player-2']:
                        import pdb; pdb.set_trace()
                    
                    players[obss["turn_player"]].model_format.append({"role": "user", "content": obss[obss["turn_player"]]})
                    turn_data = {
                        "speaker": current_player,
                        "player-1": obss['player-1'],
                        "player-2": obss['player-2'],
                    }
                    conversations_save.append(turn_data)
                    break
                
                players[current_player].model_format.append({"role": "user", "content": obss[current_player]})
                
                resample_count += 1
                refresh_count += 1
                
                players[current_player].observe(obss[current_player])
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                
                # Log the response
                #log.log(
                #    key=current_player,
                #    value=resp,
                #    title=f"generate t={t} try={resample_cnt}"
                #)  
                
                # Update environment
                #stepped = False
                

            t += 1

    # Log final results
    #import pdb; pdb.set_trace()
    for pname in players:
        players[pname].observe(obss[pname])
    open( f"count_all_end_runs_before_filter_{exp_name}.txt", "a").write(f"Running {exp_name}\n")
    if game_cls == OptimizationEnv:
        # Log the normalized reward using the same key as in evaluate.py
        reward_norm = "score_norm"
        open( f"reward_normalized_{game_cls}_{exp_name}.txt", "a").write(
            f"{reward_norm}: {obss['info'][reward_norm]}\n"
        )
    
    
    try:
        # Apply the same threshold check as in evaluate.py
        if obss['info'][reward_norm] > threshold:
            save_itinerary_data(game_cls, conversations_save, player_1_data, player_2_data, metadata, reward_normalized = obss['info'][reward_norm], output_path=f"output_{game_cls}_{exp_name}.jsonl", env=env)
            # here the metadata is the fname
    except:
        open( f"count_all_end_runs_no_normalization_{exp_name}.txt", "a").write(f"Running {exp_name}\n obss: {obss}")
        pass
    log.flush()
    for pname, player in players.items():
        log.flush_key(pname, title=f"{pname} Log")
        log.write(f"Final {pname} Prompt", player.prompt)
    
    
    result = {**obss, "t": t,
              "num_turns": len(env.game.action_log),
              "num_words": count_words(env.game.action_log)}
    log.write("Result", json.dumps(result)) #**metadata,
    log.flush()
    log.close()
    


    open( f"count_all_end_runs_after_filter_{exp_name}.txt", "a").write(f"{obss['info']}\n")


class AutoAcceptPlayer:
    """A wrapper player that automatically accepts any proposal it receives.
    
    NOTE: This class is not used in the main full_conversation_reproposal mode,
    which uses natural LLM responses from both players. Kept for potential other uses.
    """
    
    def __init__(self, base_player):
        self.base_player = base_player
        self.role = base_player.role
        self.console = base_player.console
        self.model_kwargs = getattr(base_player, 'model_kwargs', {})
        self._should_auto_accept = False
        
    @property
    def prompt(self):
        return self.base_player.prompt
        
    @prompt.setter  
    def prompt(self, value):
        self.base_player.prompt = value
    
    @property
    def api_tracker(self):
        return getattr(self.base_player, 'api_tracker', None)
        
    @api_tracker.setter
    def api_tracker(self, value):
        self.base_player.api_tracker = value
        
    def observe(self, obs):
        # Check if this observation contains a proposal
        if "Partner: [propose]" in obs or "Proposal:" in obs:
            # This is a proposal, we'll auto-accept
            self._should_auto_accept = True
        else:
            self._should_auto_accept = False
        return self.base_player.observe(obs)
    
    def respond(self, **kwargs):
        # If we should auto-accept, return an accept message
        if self._should_auto_accept:
            return "[accept] I accept this proposal."
        else:
            # Otherwise, use the base player's response
            return self.base_player.respond(**kwargs)


def main(
    exp_name: str,
    game: Optional[Literal["matching"]]="matching",
    mode: Optional[Literal["selfplay", "prompted_sp", "full_conversation_reproposal"]]="selfplay",
    new_data: Optional[bool]=True,
    resume: Optional[int]=0,
    end: Optional[int]=50,
    samples_per_game: Optional[int]=1,
    user_model_id: Optional[str]="gpt-4.1",
    agent_model_id: Optional[str]="gpt-4.1",
    dry_run: Optional[bool]=False,
    use_word_limit: Optional[bool]=False,
    track_costs: Optional[bool]=False,
    threshold: Optional[float]=0.5,
    temperature: Optional[float]=0.7,
    load_game_state: Optional[str]=None,#"/home/nickatomlin/georgiazhou/new_dialop/RL-matching/dialop/data/optimization.jsonl",
    auto_accept: Optional[bool]=False,
    use_sglang: Optional[bool]=True,
    sglang_url: Optional[str]="http://localhost:8000",
):
    
    open( f"count_all_begin_runs_{exp_name}.txt", "a").write(
        f"Running {exp_name} with {game} in {mode} mode; resume: {resume}, end: {end}, samples_per_game: {samples_per_game}, user_model_id: {user_model_id}, agent_model_id: {agent_model_id}, dry_run: {dry_run}, use_word_limit: {use_word_limit}, track_costs: {track_costs}, load_game_state: {load_game_state}\n"
    )
    
    game_cls = GAME_CLSS[game]
    EXP_DIR = RESDIR / game
    
    # --------------------------------------------------
    # Determine the list of initial game states.
    # Priority order:
    #   1. User-supplied JSONL via --load-game-state
    #   2. Legacy on-disk datasets when --new-data is False
    #   3. Empty list => random generation (current default behaviour)
    # --------------------------------------------------
    
    if load_game_state is not None:
        games = []
        with open(load_game_state) as f:
            for line in f:
                gamestate_entry = json.loads(line)
                # Check if this is a nested format with game_state or direct format
                if "game_state" in gamestate_entry:
                    # Extract the game_state and add necessary fields for compatibility
                    game_data = gamestate_entry["game_state"]
                else:
                    # Direct format like data/optimization.jsonl - use the entry as-is
                    game_data = gamestate_entry
                
                # Add missing fields that selfplay expects
                if "result" not in game_data:
                    game_data["result"] = {"norm": 0.0, "score": 0.0}
                games.append(game_data)
    elif not new_data:
        if game_cls == OptimizationEnv:
            DATA_PATH = DATADIR / "reviewer.jsonl.post"
        elif game_cls == OptimizationEnv:
            DATA_PATH = DATADIR / "optimization.jsonl.post"
        with open(DATA_PATH) as f:
            games = [json.loads(line) for line in f]
    else:
        games = []
    
    os.makedirs(EXP_DIR / exp_name, exist_ok=True)
    
    
    # Create generator for eval mode.
    if mode == "selfplay":
        gen = selfplay(game_cls, games, samples_per_game, resume, end)
    elif mode == "prompted_sp":
        gen = prompted_selfplay(game_cls, games, samples_per_game, resume, end)
    elif mode == "full_conversation_reproposal":
        gen = full_conversation_reproposal(game_cls, games, samples_per_game, resume, end)
    else:
        raise NotImplementedError()
    
    def create_players():
        # Check if metadata contains mode information  
        current_mode = mode
        current_proposer = None # Removed proposer parameter
        if metadata and isinstance(metadata, dict):
            current_mode = metadata.get("mode", mode)
            current_proposer = metadata.get("original_proposer", None) # Use original_proposer from metadata
            
        print("Initializing players...")
        # Create prompts.
        #pdb.set_trace()
        if game_cls == OptimizationEnv:
            # Load matching prompt
            prompt_path = FPATH / "dialop" / "prompts" / "matching_prompt.txt"
            if not prompt_path.exists():
                prompt_path = FPATH / "dialop" / "prompts" / "optimization.txt"
            with open(prompt_path) as f:
                matching_prompt = f.read()
            p1, p2 = "player-1", "player-2"
            p1_prompt, p2_prompt = matching_prompt, matching_prompt
        elif game_cls == OptimizationEnv:
            # if use_word_limit:
            #     with open(FPATH / "prompts" / "planning_agent_timed.txt") as f:
            #         agent_prompt = f.read()
            #     with open(FPATH / "prompts" / "planning_user_timed.txt") as f:
            #         user_prompt = f.read()
            # else:
            
            with open(FPATH / "prompts" / "planning_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "planning_user.txt") as f:
                user_prompt = f.read()
        elif game_cls == OptimizationEnv:
            with open(FPATH / "prompts" / "mediation_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user0.txt") as f:
                user0_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user1.txt") as f:
                user1_prompt = f.read()
        
        if game_cls == OptimizationEnv:
            p1, p2 = "player-1", "player-2"
            p1_prompt, p2_prompt = matching_prompt, matching_prompt
            optional1, optional2 = None, None  # No optional content for matching game
            ##optional1 = p1_prompt[
            ##    p1_prompt.index("EXAMPLE 1"):]
            ##optional1 = optional1[:optional1.index("EXAMPLE 2")]
            ##optional2 = p2_prompt[
            ##    p2_prompt.index("EXAMPLE 2"):]
            ##optional2 = optional2[:optional2.index("EXAMPLE 2")]
        elif game_cls == OptimizationEnv:
            p1, p2 = "agent", "user"
            p1_prompt, p2_prompt = agent_prompt, user_prompt
            optional1, optional2 = None, None
        elif game_cls == OptimizationEnv:
            p1, p2, p3 = "user0", "user1", "agent"
            optional = agent_prompt[agent_prompt.index("User 0 Information"):]
            optional = optional[:optional.index("\n\n") + 2]
        
        if dry_run:
            players = {p1: DryRunPlayer(p1_prompt, p1, console),
                       p2:  DryRunPlayer(p2_prompt, p2, console)}
        else:
            # Create base players - check if model is local or API-based
            def create_player(prompt, role, model_id, optional=None):
                # When using sglang, always use the sglang server and ignore model_id
                if use_sglang:
                    print(f"Using SglangModelPlayer for {role} at {sglang_url} (ignoring model_id)")
                    return SglangModelPlayer(
                        prompt,
                        role,
                        console,
                        optional=optional,
                        sglang_url=sglang_url,
                        temperature=temperature,
                    )
                # Check if this is a local model (Hugging Face style path)
                if "/" in model_id and not model_id.startswith("gpt"):
                    if HFModelPlayer is not None:
                        print(f"Using HFModelPlayer for {role} with model {model_id}")
                        return HFModelPlayer(prompt, role, console,
                                           model_path=model_id,
                                           optional=optional,
                                           temperature=temperature)
                    elif LocalModelPlayerVLLM is not None:
                        print(f"Using LocalModelPlayerVLLM for {role} with model {model_id}")
                        return LocalModelPlayerVLLM(prompt, role, console,
                                                   model_path=model_id,
                                                   optional=optional)
                    else:
                        print(f"Warning: Local model players not available, falling back to API")
                        return LLMPlayer(prompt, role, console,
                                       optional=optional,
                                       model_kwargs={'model': model_id})
                else:
                    # Use API-based player
                    return LLMPlayer(prompt, role, console,
                                   optional=optional,
                                   model_kwargs={'model': model_id})
            
            player1 = create_player(p1_prompt, p1, agent_model_id, optional1)
            player2 = create_player(p2_prompt, p2, user_model_id, optional2)
            
            # In full_conversation_reproposal mode, both players remain as LLMPlayer objects
            # The original proposer will naturally make the proposal, and the other will respond naturally
            if current_mode == "full_conversation_reproposal":
                print(f"Mode: {current_mode} - {current_proposer} will make the proposal, non-proposer will auto-accept with 'Lets think step by step. [accept]'")
                
            players = {p1: player1, p2: player2}
        return players

    def create_env():
        print("Initializing envs...")
        env = OptimizationEnv()
        if use_word_limit:
            env = ForceProposal(env, ["player-1", "player-2"])
        return env

    if dry_run:
        max_length = 15
    else:
        max_length = 35

    # Evaluate.
    times = []
    for i, (data, fname, metadata) in enumerate(gen):
        if (EXP_DIR / exp_name / f"{fname}.out").exists():
            continue
        if not dry_run and i % 20 == 1:
            mean_time = np.mean(times) if times else 0.0
            print(f"Sleeping... {mean_time:.1f}")
            time.sleep(3)
            pass
        print(fname)

        start = time.time()
        run(
            game_cls,
            data,
            metadata,
            create_players,
            create_env,
            EXP_DIR / exp_name /f"{fname}.out",
            use_word_limit=use_word_limit,
            max_length=max_length,
            exp_name=exp_name,
            threshold=threshold
        )
        elapsed = (time.time() - start) / 60
        times.append(elapsed)
        print(f" == Finished {i} {elapsed:.1f} == ")

    # Track API costs using the CostTracker from simple setting
    if track_costs:
        try:
            print("\nGenerating cost summary...")
            # Use the CostTracker from simple setting
            tracker = CostTracker(str(EXP_DIR / exp_name))
            tracker.print_summary(detailed=True)
        except Exception as e:
            print(f"Error generating cost summary: {e}")

    

    #with open("query_executor_prompt.txt", "w") as f:
    #    f.write(env.search.prompt)
    #test_queries = [
    #    "Search(fields=[name], text_query='not touristy', filters=[category == restaurant]"
    #]
    #env.search(test_queries[0])

    # Try scoring a proposal
    #proposal = "[The Dive, Saul's, NULL]"
    #proposal_info = env._propose(proposal)
    #print(proposal_info)


if __name__ == "__main__":
    tyro.cli(main)