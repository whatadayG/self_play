import json
import openai
import pdb
from openai import OpenAI

import os
import random
import pathlib
from rich.prompt import IntPrompt, Prompt
from rich.markup import escape

from .envs import DialogueEnv
from .utils import num_tokens



try:
    with open("/home/nickatomlin/georgiazhou/dialop/dialop/.api_key") as f:
        x = json.load(f)
        client = OpenAI(api_key=x["api_key"], organization=x["organization"])
    print("Loaded .api_key")
except Exception as e:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as inner_e:
        client = None
        print(inner_e)
    print(e)  
        
    print("Loaded .api_key")


if (client is None) or (not getattr(client, "api_key", None)):
    print("Warning: no OpenAI API key loaded.")


class OutOfContextError(Exception):
    pass

class DryRunPlayer:

    def __init__(self, prompt, role, console, task="planning"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.calls = 0
        self.task = task
        

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.calls += 1
        if self.role == "agent" and self.calls == 5:
            if self.task == "planning":
                return f" [propose] [Saul's, Cookies Cream, Mad Seoul]"
            elif self.task == "mediation":
                return f" [propose] User 0: [1], User 1: [15]"
        elif self.role == "user" and self.calls == 6:
            return f" [reject]"
        return f" [message] {self.calls}"

class LLMPlayer:

    def __init__(self, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", optional=None, strategy = None):
        #pdb.set_trace()
        self.count_agent_path = 0
        self.prompt = prompt
        self.user_planning_extra = "You want to accept a proposal even when it's not perfect. "
        self.user_planning_4o = "Remember to format your message with a type like '[message]' or '[think]' or '[accept]' or [reject]"
        self.user_planing_final = "You have to accept the next proposal because you are runing out of conversation time"
        self.agent_planning_extra = "Messages must be formatted with a type like '[message]' or '[tool]' or '[think]' or '[propose]'. Remember, user might not say everything they want or forget things that are important to them. It's important to ask questions to the user to understand the user's priorities so you can propose the best itinerary that satisfies the user's most important preferences."
        self.user_priority = "You should optimize for the features with high importance socres because it directly affects your final score."
        self.role = role
        
        self.console = console      
        self.model = "gpt-4.1" #not 4o
        self.optional = optional
        self.removed_optional = False
        self.strategy = None
        self.model_format = {"system": "", "user": "", "assistant": ""}
        self.model_format["system"] = prompt
        # API tracker will be attached externally (e.g., by the evaluation loop)
        # Default to None so we can safely check for it later.
        self.api_tracker = None
        if self.role in ["user", "agent", "user0", "user1"]:
            stop_tokens = ["User", "Agent", "You", "\n"]
            if self.role == "user":
                ##RO_planning_user_filepath = '/Users/georgiazhou/research_machine/dialop/dialop/RL/style_per_persona.txt'
                ##RO_planning_user_explain_filepath = '/Users/georgiazhou/research_machine/dialop/dialop/RL/explanation_per_persona.txt'
                ##with open(RO_planning_user_filepath, 'r') as f:
                ##    all_planning_personas = json.load(f)
                ##with open(RO_planning_user_explain_filepath, 'r') as f:
                ##    all_planning_personas_explain = json.load(f)
            ##
                ##persona_num = random.randint(0, len(all_planning_personas)-1)
                ##persona_tokens = all_planning_personas[str(persona_num)]
                ##tokens_list = list(persona_tokens.keys())
                ##persona_explain = {}
                ##for token in tokens_list:
                ##    persona_explain[token] = all_planning_personas_explain[token]
                ##    
                ##self.persona_tokens = str(persona_tokens)
                ##self.persona_explain = str(persona_explain)
                ##
            
                ##self.prompt =  "You are a helpful assistant. This is your communication style and their corresponding description: " + self.persona_explain + "\n" + "here is how much you score on each style attributes" + self.persona_tokens + "the scale is 0-1. 1 means the style is extremely obvious and 0 means the style is not obvious at all. Behave accordingly." +  "\n" + self.prompt
                self.user_prompt_obss = ''
                self.prompt =  self.prompt + self.user_planning_extra + " Unlike the examples above, you can only list up to 2 preferences up front." + self.user_priority
            if self.role == "agent":
                self.temp_prompt = {}
                self.prompt = self.prompt + self.agent_planning_extra
                self.strategy = strategy
        elif self.role in ["player-1", "player-2"]:
            stop_tokens = ["Partner", "You", "\n"]
        else:
            raise NotImplementedError
        self.model_kwargs = dict(
            
            model=self.model,
            #temperature=0.1,
            #top_p=.95,
            #frequency_penalty=0,
            #presence_penalty=0,
            stop=stop_tokens,
        )
        if model_kwargs is not None:
            if model_kwargs.get('temperature') is None:
                
                self.model_kwargs.update(**model_kwargs)
            else:
                import pdb; pdb.set_trace()
        self.prefix = prefix
    #    self.model = "gpt-3.5-turbo"

    def observe(self, obs, ignore_obs = False, temporal = None, divident = None, info = False):
        #pdb.set_trace()
        if self.role == "user":
            self.user_prompt_obss += (self.prefix + obs)
        
        if temporal:
            if divident is None:
                num = 10
            else:
                num = divident
                if self.count_agent_path == 0:
                    self.temp_prompt = {}
                if self.count_agent_path < num:
                    self.temp_prompt[self.count_agent_path] = str(self.prompt) + '\n' "--special prompt--"+ '\n' 
                    self.temp_prompt[self.count_agent_path] += obs
                    self.count_agent_path += 1
                else:
                    self.temp_prompt[self.count_agent_path % num] += ( '\n' + obs)
                    self.count_agent_path += 1
        elif ignore_obs:
            pass 
        else:
            if info:
                self.prompt += (obs + "\n" + "Here is the start of your current conversation:\n")
            else:
                if "Partner:" in obs:
                    self.prompt += obs

                else:
                    self.prompt += (self.prefix + obs)
            


    def respond_ready_user_planning(self, extracted_features):
        self.console.rule(f"{self.role}'s turn")
        if not extracted_features:
            pdb.set_trace()
            #file_path = '/Users/georgiazhou/research_machine/dialop/RL_data/20_persona/'
        known_features_str = str(extracted_features)
        all_users = {}
        self.prompt = self.prompt + known_features_str
        for i in range(5):
            response = self.respond(vary = True)
            all_users[i] = response
        return all_users
    
                
    def respond(self, t = 0, max_len = 3, vary = False, propose = False, temporal_id = None, strategy = None):
        
        #pdb.set_trace()
        
                
                
        
        

        
        if temporal_id:
            #pdb.set_trace()
            try:
                selfprompt = self.temp_prompt[temporal_id]
                if propose:
                    selfprompt += 'You must make a proposal now.'
            except:
                #pdb.set_trace()
                print(f"Warning: No prompt found for temporal index {temporal_id}")
                selfprompt = self.prompt
        elif propose:
            selfprompt = 'You must make a proposal now.' + self.prompt + 'You must make a proposal now.'
            #print(selfprompt)
        else:
            if strategy:
                selfprompt = self.prompt + '\n' + 'Here is your conversational strategy: ' + strategy
            else:
                # there was a reason that it's this way; code modification logistic stuff 
                if self.strategy:
                    selfprompt = self.prompt + '\n' + 'Here is your conversational strategy: ' + self.strategy
                else:
                    selfprompt = self.prompt
        
        
            
        
        if not selfprompt.endswith(self.prefix):
            if propose:
                selfprompt += (self.prefix + '[propose]')
            else:
                if self.role == "user":
                    if t>= (max_len - 5):
                        selfprompt += self.user_planing_final

                    ##else:
                    ##    prompt += self.user_planning_extra
                ##if self.role == "agent":
                ##    prompt += self.agent_planning_extra

                selfprompt += self.prefix
        
        #self.console.print(escape(prompt))
        #remaining = 4096 - num_tokens(selfprompt)
        #if remaining < 0 and self.optional:
        #    self._remove_optional_context()
        #    remaining = 4096 - num_tokens(selfprompt)
        # Still out of context after removing
        #if remaining < 0:
        #    print("OUT OF CONTEXT! Remaining ", remaining)
        #    raise OutOfContextError()
        prompt = {'role': 'system', 'content': selfprompt,}

        if vary:
            kwargs = dict(**self.model_kwargs)
            # Then override with variation settings
            kwargs.update(
                temperature = 1.8,
                messages = [prompt],
                #max_tokens = remaining,
                seed = random.randint(1, 10000)
            )
        else: 
            kwargs = dict(
                messages = [prompt],
                #max_tokens= remaining,
                #  it was min 128 and remianing
            )
            kwargs.update(**self.model_kwargs)
        # 'message' key added for new openai API; role is the role of the mssage
        del kwargs['stop']
        #import pdb; pdb.set_trace()
        #if kwargs.get('model') != 'gpt-4.1':
        #    import pdb; pdb.set_trace()
        if client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY or provide .api_key when using LLMPlayer.")
        response = client.chat.completions.create(**kwargs)
        self.console.print("player: ", self.role)
        self.console.print("Response: ",
                           escape(response.choices[0].message.content.strip()))
        self.console.print("stop: ", response.choices[0].finish_reason)
        if response.choices[0].finish_reason == "length":
            if not self.optional:
                import pdb; pdb.set_trace()
                raise OutOfContextError()
            self._remove_optional_context()
            response = client.chat.completions.create(**kwargs)
            self.console.print("Response: ",
                               escape(response.choices[0].message.content.strip()))
            self.console.print("stop: ", response.choices[0].finish_reason)
        self.console.print(response.usage)

        # ------------------------------------------------------------------
        # Cost tracking: log this API call if a tracker is attached.
        # The evaluation script attaches an `OpenAITracker` instance to each
        # player via `player.api_tracker`.  We record the raw prompt that was
        # sent (selfprompt) and the assistant response text.
        # ------------------------------------------------------------------
        if getattr(self, "api_tracker", None) is not None:
            try:
                self.api_tracker.log_api_call(
                    model=self.model_kwargs.get("model", self.model),
                    input_text=selfprompt,
                    output_text=response.choices[0].message.content.strip(),
                    project_id=self.role,
                    endpoint="chat.completions"
                )
            except Exception as e:
                # Logging failures should not break the dialogue loop.
                self.console.print(f"[bold red]Error logging API call:[/bold red] {e}")

        return response.choices[0].message.content.strip()

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True

class HumanPlayer:

    def __init__(self, prompt, role, console, prefix="\nYou:"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.prefix = prefix

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        self.console.rule(f"Your turn ({self.role})")
        #self.console.print(escape(self.prompt))
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
