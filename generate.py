"""
Email Generation Module

This module provides the GenerateEmail class for calling Azure OpenAI's ChatCompletions API
to generate variations of emails (shorten, lengthen, change tone).

It uses prompts defined in prompts.yaml and communicates with Azure OpenAI via the official
OpenAI Python SDK configured with Azure-specific endpoints.

Configuration:
    - OPENAI_API_BASE: Azure endpoint URL (from .env)
    - OPENAI_API_KEY: Azure API key (from .env)
    - Model: Deployment name (passed to __init__)
    - Prompts: Located in prompts.yaml

See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/chatgpt
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml

# Load environment variables from .env file
load_dotenv()

# Load prompt templates from YAML file
with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)


class GenerateEmail:
    """Generate email variations using Azure OpenAI ChatCompletions API.
    
    This class wraps the OpenAI ChatCompletions API to provide email generation
    capabilities. It supports three main operations:
    - Shorten: Condense emails while preserving key information
    - Lengthen: Expand emails with additional context and clarity
    - Tone: Rewrite emails in different tones (friendly, sympathetic, professional)
    
    Attributes:
        client (OpenAI): OpenAI client configured for Azure endpoints.
        deployment_name (str): Name of the Azure deployment to use for API calls.
    """

    def __init__(self, model: str):
        """Initialize the GenerateEmail client.
        
        Args:
            model (str): Azure deployment name (e.g., 'gpt-4.1', 'gpt-4o-mini').
        
        Raises:
            ValueError: If OPENAI_API_BASE or OPENAI_API_KEY environment variables
                       are not set in the .env file.
        """
        # Initialize Azure OpenAI client with credentials from environment
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self, messages: list) -> str:
        """Call Azure OpenAI ChatCompletions API and extract response text.
        
        This is the core method that communicates with the Azure OpenAI ChatCompletions
        endpoint. It sends a list of messages and returns the model's response.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content' keys.
                           Example: [
                               {"role": "system", "content": "You are a helpful assistant."},
                               {"role": "user", "content": "Shorten this email: ..."}
                           ]
        
        Returns:
            str: The text content of the model's response, or None if the API call fails.
        
        Raises:
            Exception: Caught and logged; returns None instead of propagating.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
            )
            # Extract the response text from the first choice
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None

    def get_prompt(self, prompt_name: str, prompt_type: str = 'user', **kwargs) -> str:
        """Retrieve and format a prompt template from prompts.yaml.
        
        Loads a prompt template from the YAML file and formats it with provided
        keyword arguments. Supports both 'system' and 'user' prompt types.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., 'shorten', 'lengthen', 'tone').
            prompt_type (str, optional): Type of prompt ('system' or 'user'). Defaults to 'user'.
            **kwargs: Format arguments for the prompt template (e.g., selected_text, tone).
        
        Returns:
            str: Formatted prompt string with all placeholders filled.
        
        Raises:
            KeyError: If prompt_name or prompt_type not found in prompts.yaml.
        """
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)

    def send_prompt(self, user_prompt: str, system_msg: str = "You are a helpful assistant.") -> str:
        """Send a system and user message pair to the API.
        
        Constructs a message list with system and user roles and sends it to the API.
        
        Args:
            user_prompt (str): The user's prompt/request.
            system_msg (str, optional): System prompt to set the model's behavior.
                                       Defaults to "You are a helpful assistant."
        
        Returns:
            str: The model's response text, or None if the API call fails.
        """
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)

    def judge(self, metric: str, original_text: str, edited_text: str):
        """Run a judge prompt for the given metric.

        Args:
            metric: One of 'faithfulness', 'completeness', 'conciseness'.
            original_text: The original email text.
            edited_text: The edited/model-generated email text.

        Returns:
            Parsed JSON dict if available, otherwise raw string response or None.
        """
        metric_map = {
            "faithfulness": "faithfulness_judge",
            "completeness": "completeness_judge",
            "conciseness": "conciseness_judge",
        }
        prompt_name = metric_map.get(metric)
        if not prompt_name:
            raise ValueError(f"Unsupported metric: {metric}")

        args = {"selected_text": original_text, "model_response": edited_text}
        system_prompt = self.get_prompt(prompt_name, prompt_type="system", **args)
        user_prompt = self.get_prompt(prompt_name, prompt_type="user", **args)
        raw = self.send_prompt(user_prompt, system_prompt)

        # Try to parse JSON per the expected format
        if raw is None:
            return None
        try:
            import json
            return json.loads(raw)
        except Exception:
            return raw

    def generate(self, action: str, email_content: str, tone: str = None) -> str:
        """Generate email modifications based on the specified action.
        
        Main public method for generating email variations. Loads appropriate prompts
        based on the action type and calls the API.
        
        Args:
            action (str): Type of modification ('shorten', 'lengthen', or 'tone').
            email_content (str): The original email text to be modified.
            tone (str, optional): Desired tone for the 'tone' action.
                                 Valid values: 'friendly', 'sympathetic', 'professional'.
                                 Required when action='tone', ignored otherwise.
        
        Returns:
            str: The generated/modified email text, or None if the API call fails.
        
        Examples:
            >>> generator = GenerateEmail(model='gpt-4.1')
            >>> result = generator.generate('shorten', 'Long email text here...')
            >>> result = generator.generate('tone', 'Email text...', tone='friendly')
        
        Raises:
            ValueError: If action is not one of the supported types.
        """
        args = {"selected_text": email_content}

        if action == "shorten":
            # Load prompts for shortening action
            system_prompt = self.get_prompt('shorten', prompt_type='system', **args)
            user_prompt = self.get_prompt('shorten', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        elif action == "lengthen":
            # Load prompts for lengthening/elaborating action
            system_prompt = self.get_prompt('lengthen', prompt_type='system', **args)
            user_prompt = self.get_prompt('lengthen', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        elif action == "tone":
            # Load prompts for tone change action, include tone in template args
            args["tone"] = tone
            system_prompt = self.get_prompt('tone', prompt_type='system', **args)
            user_prompt = self.get_prompt('tone', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        else:
            print(f"Unknown action: {action}")
            return None
