from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

class GenerateEmail():    
    def __init__(self, model: str):
        # initialize client once
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self, messages):
        """Call OpenAI ChatCompletions API and return the response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
        
    
    def get_prompt(self, prompt_name, prompt_type='user', **kwargs):
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)
    
    def send_prompt(self, user_prompt: str, system_msg="You are a helpful assistant."):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)
    
    def generate(self, action: str, email_content: str, tone: str = None) -> str:
        """Generate email modifications based on action.
        
        Args:
            action: 'shorten', 'lengthen', or 'tone'
            email_content: The email text to modify
            tone: Tone option for 'tone' action ('friendly', 'sympathetic', 'professional')
        
        Returns:
            Modified email text or None if API fails
        """
        args = {"selected_text": email_content}
        
        if action == "shorten":
            system_prompt = self.get_prompt('shorten', prompt_type='system', **args)
            user_prompt = self.get_prompt('shorten', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response
        
        elif action == "lengthen":
            system_prompt = self.get_prompt('lengthen', prompt_type='system', **args)
            user_prompt = self.get_prompt('lengthen', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response
        
        elif action == "tone":
            args["tone"] = tone
            system_prompt = self.get_prompt('tone', prompt_type='system', **args)
            user_prompt = self.get_prompt('tone', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response
        
        else:
            print(f"Unknown action: {action}")
            return None