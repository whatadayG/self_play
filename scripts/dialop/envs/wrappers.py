class WordLimit:

    def __init__(self, env, players):
        self.env = env
        self.players = players

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, word_limit, game_state=None):
        obs = self.env.reset(game_state)
        assert word_limit > 0
        self.word_limit = word_limit
        self.words_left = word_limit
        self.end_soon = False
        self.proposal_made = False
        for msg in self.game.action_log:
            if msg["type"] == "message":
                self.words_left -= len(msg["message"]["data"].split(" "))
        if obs["turn_player"] in self.players:
            obs[obs["turn_player"]] = self._insert_word_limit(
                obs[obs["turn_player"]])
        return obs

    def step(self, message):
        obss, resample = self.env.step(message)
        if "[message]" in message:
            text = message[message.index("] ") + 2:]
            self.words_left -= len(text.split(" "))
        if self.end_soon:
            if self.proposal_made and obss["turn_player"] not in self.players:
                # Accept automatically
                obss, resample = self.env.step(" [accept]")
                if not obss["done"]:
                    import pdb; pdb.set_trace()
                obss["info"]["word_limited"] = True
                return obss, resample

            if "[propose]" in message:
                self.proposal_made = self.game.is_full_proposal
        if obss["turn_player"] not in self.players:
            return obss, resample
        ob = obss[obss["turn_player"]]
        obss[obss["turn_player"]] = self._insert_word_limit(ob)
        return obss, resample

    def _insert_word_limit(self, ob):
        ob = ob.rsplit("\n", 1)
        msg = f"Words left: {self.words_left}"
        if self.words_left < 25:
            msg += "\nYou must make your best final proposal now."
            self.end_soon = True
        ob = f"{ob[0]}\n{msg}\n{ob[1]}"
        return ob


class ForceProposal:
    def __init__(self, env, players):
        self.env = env
        self.players = players

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, word_limit, game_state=None):
        obs = self.env.reset(game_state)
        assert word_limit > 0
        self.word_limit = word_limit
        self.words_left = word_limit
        self.end_soon = False
        self.proposal_made = False
        for msg in self.game.action_log:
            if msg["type"] == "message":
                self.words_left -= len(msg["message"]["data"].split(" "))
        if obs["turn_player"] in self.players:
            obs[obs["turn_player"]] = self._insert_word_limit(
                obs[obs["turn_player"]])
        return obs

    def step(self, message):
        # Run the underlying env one step first
        obss, resample = self.env.step(message)

        # Track word usage for the (optional) word-limit mechanic kept from the
        # original implementation.
        if "[message]" in message:
            text = message[message.index("] ") + 2:]
            self.words_left -= len(text.split(" "))

        # If we have entered the “force proposal” phase (end_soon == True)
        # we enforce a strict two-step sequence:
        #   1) current player MUST make a proposal (self.proposal_made flag)
        #   2) turn switches – the new current player MUST [accept] or [reject]
        #      that proposal.  Any other output is treated as invalid and
        #      triggers a resample by the caller.
        if self.end_soon:
            lowered_msg = message.lower()

            # -----------------------------------------------------------------
            # STATE 1 – expecting a proposal
            # -----------------------------------------------------------------
            if not self.proposal_made:
                if "[propose]" in lowered_msg:
                    # A proposal has just been made; remember this so that the
                    # next speaker will be forced to accept/reject it.
                    self.proposal_made = True
                else:
                    # The current speaker failed to propose – signal a bad turn
                    # by requesting a resample.  We simply return here; caller
                    # (evaluate_opt) will handle the resample loop.
                    return obss, True

            # -----------------------------------------------------------------
            # STATE 2 – proposal already exists; expecting accept / reject
            # -----------------------------------------------------------------
            else:
                if "[accept]" in lowered_msg:
                    # Normal env behaviour is fine; nothing extra to enforce.
                    pass
                elif "[reject]" in lowered_msg:
                    # Force immediate termination with zero reward / score.
                    obss["done"] = True
                    # Ensure we tag the info dict so downstream scripts know
                    # it was forced-end rejection.
                    if "info" not in obss:
                        obss["info"] = {}
                    obss["info"].update({
                        "forced_reject": True,
                        "score": 0,
                        "score_norm": 0.0,
                    })
                else:
                    # Any other tag is invalid – ask caller to resample.
                    return obss, True

        # Always (re-)insert the word-limit reminder for the next speaker
        try:
            turn_player = obss["turn_player"]
            if turn_player in self.players and turn_player in obss:
                obss[turn_player] = self._insert_word_limit(obss[turn_player])
        except Exception as e:
            # Log the error but continue with raw observation
            # This is defensive to prevent wrapper bugs from killing games,
            # but we log loudly so issues don't go unnoticed
            import traceback
            print(f"!!! WARNING: ForceProposal wrapper failed to insert word limit")
            print(f"!!! Exception: {e}")
            print(f"!!! {traceback.format_exc()}")
            # Fall back to raw obs without word limit

        return obss, resample

    def _insert_word_limit(self, ob):
        ob = ob.rsplit("\n", 1)
        # msg = f"Words left: {self.words_left}"
        msg = ""
        if self.words_left < 25:
            msg += "\nYou must make your best final proposal now."
            self.end_soon = True
        ob = f"{ob[0]}\n{msg}\n{ob[1]}"
        return ob

class AsymmetricForceProposal:
    def __init__(self, env, players):
        self.env = env
        self.players = players

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, word_limit, game_state=None):
        obs = self.env.reset(game_state)
        assert word_limit > 0
        self.word_limit = word_limit
        self.words_left = word_limit
        self.end_soon = False
        self.proposal_made = False
        for msg in self.game.action_log:
            if msg["type"] == "message":
                self.words_left -= len(msg["message"]["data"].split(" "))
        if obs["turn_player"] in self.players:
            obs[obs["turn_player"]] = self._insert_word_limit(
                obs[obs["turn_player"]])
        return obs

    def step(self, message):
        obss, resample = self.env.step(message)
        if "[message]" in message:
            text = message[message.index("] ") + 2:]
            self.words_left -= len(text.split(" "))
        if self.end_soon:
            if self.proposal_made and obss["turn_player"] not in self.players:
                # Accept automatically
                obss, resample = self.env.step(" [accept]")
                if not obss["done"]:
                    import pdb; pdb.set_trace()
                obss["info"]["word_limited"] = True
                return obss, resample

            if "[propose]" in message:
                self.proposal_made = self.game.is_full_proposal
            # elif self.proposal_made and "[accept]" not in message:
            #     raise self.env.game_error("Agent must [accept] the proposal")
            # else:
            #     raise self.env.game_error("Agent must send a message beginning with [propose]")
        if obss["turn_player"] not in self.players:
            return obss, resample
        ob = obss[obss["turn_player"]]
        obss[obss["turn_player"]] = self._insert_word_limit(ob)
        return obss, resample

    def _insert_word_limit(self, ob):
        ob = ob.rsplit("\n", 1)
        # msg = f"Words left: {self.words_left}"
        msg = ""
        if self.words_left < 25:
            msg += "\nYou must make your best final proposal now."
            self.end_soon = True
        ob = f"{ob[0]}\n{msg}\n{ob[1]}"
        return ob