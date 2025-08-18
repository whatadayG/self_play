import traceback
import re
from pathlib import Path

from dialop.envs import DialogueEnv, GameError
from dialop.games.optimization import OptimizationGame
from dialop.templates import OptimizationPromptTemplate

class OptimizationEnv(DialogueEnv):

    def __init__(self, one_player=False):
        self.one_player = one_player
        self.players = ["player-1", "player-2"]
        instrs = (Path(__file__).parent / "data/optimization.txt").read_text()
        self.instructions = [instrs, instrs]

    def reset(self, game_state=None):
        if game_state is not None:
            self.game = OptimizationGame.create_from_game_state(game_state, one_player=self.one_player)
            self.game.action_log = game_state["action_log"]
        else:
            self.game = OptimizationGame({}, one_player=self.one_player)
            self.game.reset()
        # Compute score range
        self.best_score = self.game.best_assignment_reward
        self.num_msgs = 0
        obss = self._init_from_action_log()
        return {**obss,
                "turn_player": self.players[self.game.turn_player],
                "done": False}

    def step(
        self,
        message,
        last_message: bool = False,
        ready: bool = False,
        pause_turn: bool = False,
        propose: bool = False,
        user_think: bool = False,
        agent_pending: bool = False,
        **kwargs,
    ):
        done = False
        reward = 0
        info = {"num_msgs": self.num_msgs}
        player = self.players[self.game.turn_player]
        # Begin thinking-step validation similar to PlanningEnv
        think_required = False
        if player == "player-2":
            # Always require a thinking step from player-2 (acts like the agent)
            think_required = True
        elif player == "player-1" and user_think:
            # Require a thinking step for player-1 only when user_think flag is set
            think_required = True
        
        if message.endswith("...") or message.endswith(":"):
            error_msg = (
                f"\nIf you output something similary to <Here are some of the highest reviewer-paper matches I see:>, you must specify WHAT the matches are in the same output. If you don't, your turn will end and your partner won't know the matches you were talking about. You must not end the message half way. Your output was: '{message}'\n"
            )
            return {
                "done": False,
                "turn_player": player,
                "player-1": error_msg if player == 'player-1' else '',
                "player-2": error_msg if player == 'player-2' else '',
                "message_type": "None"
            }, True

        if think_required:
            # Look for the first occurrence of a valid message type tag
            tag_match = re.search(r"\[(message|propose|accept|reject)\]", message, re.IGNORECASE)
            if tag_match:
                message_type = tag_match.group(1).lower()
                think_part = message[:tag_match.start()].strip()

                # Validate the thinking step content
                if think_part == "" or ("let's think step by step" not in think_part.lower() and "lets think step by step" not in think_part.lower()):
                    error_msg = (
                        f"\nError: You must output 'Let's think step by step' first before outputting '[{message_type}]<message>'. "
                        "You must show the thinking process and output the message all in the same output. "
                        f"The thinking step is empty or incorrect in your current output. Your output was: '{message}'\n"
                    )
                    return {
                        "done": False,
                        "turn_player": player,
                        "player-1": error_msg if player == 'player-1' else '',
                        "player-2": error_msg if player == 'player-2' else '',
                        "message_type": "None"
                    }, True

                # No role-based restriction: the current turn player (the recipient of the proposal)
                # is allowed to accept or reject. Validation of whether an accept/reject is permitted
                # (i.e., a full proposal exists) is handled elsewhere by `_parse_message` and
                # `_proposal_response`.
            else:
                # No valid message tag found
                error_msg = (
                    f"\nError: Your response must contain one of the following message types explicitly stated: [message], [propose], [accept], or [reject]. "
                    f"Your output was: {message}\n"
                )
                return {
                    "done": False,
                    "turn_player": player,
                    "player-1": error_msg if player == 'player-1' else '',
                    "player-2": error_msg if player == 'player-2' else '',
                    "message_type": "None"
                }, True
        # End thinking-step validation

        # Always parse only the visible part (from the first tag onwards)
        tag_for_parse = re.search(r"\[(message|propose|accept|reject)\]", message, re.IGNORECASE)
        if tag_for_parse is None:
            # Should be caught earlier, but safeguard here as well
            return {
                "done": False,
                "turn_player": player,
                "player-1": "\nError: Your message must contain a tag like [message], [propose], [accept], or [reject].",
                "player-2": "\nError: Your message must contain a tag like [message], [propose], [accept], or [reject].",
                "message_type": "None"
            }, True

        msg_for_parse = message[tag_for_parse.start():].lstrip()
        try:
            # When `propose` flag is passed (used by force_proposal scheduler),
            # treat the raw `message` as a proposal even if it is not explicitly
            # tagged with "[propose]". This mirrors how PlanningEnv handles the
            # same functionality.
            m = self._parse_message(
                    msg_for_parse,
                    edited_prompt_propose = propose,
                    can_propose = True,
                    can_respond = True,
                    must_respond = (self.game.proposal is not None),
                    )
            type_ = m["mtype"]
            content = m["msg"]
            # Enforce that when `propose` flag is True, the parsed message must be of type "propose".
            if propose and type_ != "propose":
                raise ValueError("When `propose=True`, the message must be of type 'propose'.")
            if type_ == "message":
                self.num_msgs += 1

                # Extract the visible part for the partner (everything starting from the tag)
                tag_match_visible = re.search(r"\[(message|propose|accept|reject)\]", message, re.IGNORECASE)
                message_part = message[tag_match_visible.start():] if tag_match_visible else message

                if player == "player-1":
                    obss = [message, f"\nPartner: {message_part}"]
                else:
                    obss = [f"\nPartner: {message_part}", message]

                # Log only the public portion of the message to the game history
                self.game.message({
                        "data": content,  # content is already the text after the tag
                        "from_player": self.game.turn_player,
                        "type": "utterance",
                })
            elif type_ == "propose":
                self.num_msgs += 1

                # Update game state with the proposal (content does not include the tag)
                if "Proposal:" in content:
                    proposal_content = "Proposal:" + content.split("Proposal:")[1].strip()
                else:
                    proposal_content = content
                _ = self._propose(proposal_content)

                # Build observations where the partner only sees from the tag onwards
                tag_match_visible = re.search(r"\[propose\]", message, re.IGNORECASE)
                message_part = message[tag_match_visible.start():] if tag_match_visible else message

                if player == "player-1":
                    a_obs = message  # full message with thinking for proposer
                    u_obs = f"\nPartner: {message_part}\nYou can output one of these choices: [accept] <your message to your partner> or [reject] <your message to your partner>"
                    obss = [a_obs, u_obs]  # index 0 -> player-1, 1 -> player-2
                else:
                    a_obs = message
                    u_obs = f"\nPartner: {message_part}\nYou can output one of these choices: [accept] <your message to your partner> or [reject] <your message to your partner>"
                    obss = [u_obs, a_obs]
            elif type_ == "accept" or type_ == "reject":
                #import pdb; pdb.set_trace()
                self.num_msgs += 1
                # If `last_message` is True, automatically interpret the user's
                # final act as an acceptance. This behaviour is used by
                # wrappers that terminate the dialogue early.
                ##if last_message:
                ##    type_ = "accept"

                
                obss = ["", ""]
                obss[self.game.turn_player] = f"{message}"
                obss[1 - self.game.turn_player] = f"\nPartner: [{type_}]{content}"
                done, game_infos = self._proposal_response(
                    type_ == "accept",
                    self.game.turn_player)
                info.update({
                    "score": self.game.proposal_reward,
                    "score_norm": self.game.proposal_reward / self.game.best_assignment_reward,
                })
            else:
                raise ValueError(f"Message type not found for: {message}.")
        except GameError as e:
            obss = ["", ""]
            obss[self.game.turn_player] = f"{message}\nError: {str(e)}"
            obss = {self.players[i]: obs for i, obs in enumerate(obss)}
            obss.update({
                "turn_player": self.players[self.game.turn_player],
                "done": done,
                "reward": reward,
                "info": info,
            })
            return obss, True
        except Exception as e:
            print(f"!!! {traceback.format_exc()}")
            return {
                **{p: "error" for p in self.players},
                "done": False,
                "reward": 0,
                "turn_player": self.players[self.game.turn_player]
            }, True
        obss = {self.players[i]: obs for i, obs in enumerate(obss)}
        obss.update({
            "turn_player": self.players[self.game.turn_player],
            "done": done,
            "reward": reward,
            "info": info,
        })
        return obss, False

    def _propose(self, message):
        # Parse the formal proposal and register it in the game

        proposal = self._parse_proposal(message)
        self.game.propose(None, self.game.turn_player, proposal_ids=proposal)
        proposer = 1 - self.game.turn_player
        obss = ["", ""]
        obss[proposer] = message
        obss[self.game.turn_player] = (
            f"\nPartner: {message}"
            f"\nYou can output one of these choices: [accept] <your message to your partner> or [reject] <your message to your partner>")
        return obss

    def _init_from_action_log(self):
        obss = {}
        for i, player in enumerate(self.players):
            if self.one_player:
                # In one-player mode, both players see the same unscaled, complete table
                table = "\n".join([",".join(map(str, row)) for row in self.game.unscaled_table])
            else:
                # In two-player mode, each player sees their own scaled table
                table = "\n".join([",".join(map(str, row)) for row in self.game.tables[i]])
            obss[player] = OptimizationPromptTemplate.render(
                table=table,
                messages=self.game.action_log,
                player_id=i,
                any=any,
            ).rstrip()
        return obss

    def _word_overlap_search(self, options, item):
        import string
        translator = str.maketrans('', '', string.punctuation)

        def normalize(text):
            return [w.translate(translator).lower() for w in text.split()]

        item_words = set(normalize(item))

        best_option = None
        best_score = -1
        for option in options:
            option_words = normalize(option)
            overlap = len(item_words.intersection(option_words))
            if overlap > best_score:
                best_score = overlap
                best_option = option

        # If no word overlap is found, treat this as an invalid (or incomplete)
        # formal proposal rather than silently defaulting to an arbitrary
        # reviewer/paper.  We want to force the player to either:
        #   1. Provide a *complete* unambiguous mapping in the proposal, **or**
        #   2. Discuss any uncertainty in a separate [message] before sending a
        #      formal [propose].

        if best_score == 0:
            # Nothing in `item` matched any canonical option.
            raise GameError(
                (
                    "Unable to match '{item}' to a valid name. Your output after 'Proposal:' must "
                    "list one reviewer for every paper with no placeholders or open "
                    "questions. If you still need to discuss or negotiate, send a "
                    "[message] first or include your uncertainty *before* the formal "
                    "proposal section. So your output would look like [propose] <anything you want to say to your partner> Proposal: <your proposal here>."
                )
            )

        return best_option

    def _parse_proposal(self, message):
        pattern = r"\n|<br/>"
        proposal_lines = re.split(pattern, message)
        proposal_lines = [line.strip() for line in proposal_lines]
        proposal_lines = [line for line in proposal_lines if line]
        # Validate header line
        header_line = proposal_lines[0]
        if header_line.strip().startswith("Proposal:"):
            # Normalize header to exactly 'Proposal:' and discard any trailing content on the same line
            proposal_lines[0] = "Proposal:"
        else:
            raise GameError(
                "Your proposal after [propose] <anything you want to say to your partner> must start with 'Proposal:' (capital P, trailing colon)."
            )

        # Validate that the correct number of assignment lines are present
        expected_assignments = self.game.num_rows  # number of papers/reviewers in the game (default 8)
        provided_assignments = len(proposal_lines) - 1  # exclude the header line
        if provided_assignments != expected_assignments:
            raise GameError(
                f"Your proposal must list exactly {expected_assignments} paper–reviewer assignments (one per paper). "
                f"I detected {provided_assignments}. Example format:\n"
                "Proposal:<br/>&emsp; - PAPER_1: REVIEWER_A<br/>&emsp; - PAPER_2: REVIEWER_B<br/>&emsp; ..."
                "Also remember that don't put any message after Proposal:. You can only put your proposal after Proposal:, but you can put a message between [propose] and Proposal:"
            )
        # Use the original (unmutated) lists so that indices correspond to the
        # true row/column numbers in the table.
        workers_full = self.game.WORKERS[:self.game.num_rows]
        tasks_full = self.game.TASKS[:self.game.num_cols]

        used_workers = set()
        used_tasks = set()
        parsed_proposal_indices = []  # (reviewer_idx, paper_idx)

        # Skip the header line ("Proposal:") when iterating over assignment lines
        for line in proposal_lines[1:]:
            # Remove any leading list markers like '-'
            cleaned_line = line.lstrip("- ")
            # Split on the *last* colon so that colons inside the paper title are preserved
            if ":" not in cleaned_line:
                raise GameError(
                    "Each line of your proposal must be of the form Paper: Reviewer"
                )
            task_part, worker_part = cleaned_line.rsplit(":", 1)
            task_part = task_part.strip()
            worker_part = worker_part.strip()

            # Normalize and match to canonical names
            task = self._word_overlap_search(tasks_full, task_part)
            worker = self._word_overlap_search(workers_full, worker_part)

            task_index = tasks_full.index(task)
            worker_index = workers_full.index(worker)

            # Ensure each paper and reviewer appears at most once in the proposal
            if task_index in used_tasks:
                raise GameError(f"Paper '{task}' is listed more than once in your proposal.")
            if worker_index in used_workers:
                raise GameError(f"Reviewer '{worker}' is assigned to multiple papers in your proposal.")

            used_tasks.add(task_index)
            used_workers.add(worker_index)

            parsed_proposal_indices.append((worker_index, task_index))

        return parsed_proposal_indices