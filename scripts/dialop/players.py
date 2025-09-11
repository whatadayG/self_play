"""
Legacy player classes for backward compatibility.
Imports and re-exports the unified player implementations.
"""

from rich.prompt import IntPrompt, Prompt
from rich.markup import escape

# Import and re-export unified players for compatibility
from .openai_model_player import OpenAIModelPlayer as LLMPlayer
# Note: This provides basic compatibility but some features may differ


class OutOfContextError(Exception):
    """Raised when context window is exceeded."""
    pass


class HumanPlayer:
    """Interactive player that prompts for human input."""
    
    def __init__(self, prompt, role, console, prefix="\nYou:"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.prefix = prefix

    def observe(self, obs):
        """Add observation to prompt."""
        self.prompt += obs

    def respond(self):
        """Get response from human via console prompts."""
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        self.console.rule(f"Your turn ({self.role})")
        
        resp = ""
        if self.prefix.strip().endswith("You to"):
            id_ = Prompt.ask(
                escape(f"Choose a player to talk to"),
                choices=["0","1","all"])
            resp += f" {id_}:"
            
        mtypes = ["[message]", "[propose]", "[accept]", "[reject]"]
        choices = " ".join(
                [f"({i}): {type_}" for i, type_ in enumerate(mtypes)])
        type_ = IntPrompt.ask(
                escape(
                    f"Choose one of the following message types:"
                    f"\n{choices}"),
                choices=["0","1","2","3"])
        message_type = mtypes[type_]
        
        if message_type not in ("[accept]", "[reject]"):
            content = Prompt.ask(escape(f"{message_type}"))
        else:
            content = ""
            
        resp += f" {message_type} {content}"
        return resp


class DryRunPlayer:
    """Mock player for testing with predetermined responses."""
    
    def __init__(self, prompt, role, console, task="planning"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.calls = 0
        self.task = task

    def observe(self, obs):
        """Add observation to prompt."""
        self.prompt += obs

    def respond(self):
        """Return predetermined responses based on call count."""
        self.calls += 1
        if self.role == "agent" and self.calls == 5:
            if self.task == "planning":
                return f" [propose] [Saul's, Cookies Cream, Mad Seoul]"
            elif self.task == "mediation":
                return f" [propose] User 0: [1], User 1: [15]"
        elif self.role == "user" and self.calls == 6:
            return f" [reject]"
        return f" [message] {self.calls}"