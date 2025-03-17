import requests
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic

# Remove redundant client classes since we have LLMClient
class ClaudeClient:
    def __init__(self, model_name="claude-3-sonnet-20240229"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error calling Claude: {str(e)}")
    
def call_gpt4(client, input_text, logger):
    try:
        response = client.chat.completions.create(
            model=client.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        output = response.choices[0].message.content
        logger.info(f"GPT-4 Input: {input_text}")
        logger.info(f"GPT-4 Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error calling GPT-4: {str(e)}")
        logger.info(f"Input, {input_text}")
        return f"Error: {str(e)}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_model_output(model_type, model, input_text, logger, defenses=None):
    if isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0], str):
        input_text = input_text[0]
    elif isinstance(input_text, dict):
        input_text = next(iter(input_text.values()))
    
    if model_type == "gpt":
        return call_gpt4(model, input_text, logger)
    elif model_type == "together":
        print("input_text", input_text)
        return model.generate(input_text)
    elif model_type == "claude":
        # Create a ClaudeClient instance with the correct model name
        claude_client = ClaudeClient(model_name="claude-3-sonnet-20240229")
        return claude_client.generate(input_text)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

import os
import json
import logging
from typing import Any, Dict
from openai import OpenAI

def setup_logging(args):
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    print(args)
    
    log_file = os.path.join(log_dir, f"{args['log_file']}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
import json
import logging
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from llm_client import LLMClient

def setup_logging(args):
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{args['log_file']}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_model_output(model: LLMClient, input_text: str, logger, defenses: Any = None) -> str:
    """
    Uses the provided LLMClient to generate a response for the given input_text.
    Optional 'defenses' parameter is included for future expansions.
    """
    if isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0], str):
        input_text = input_text[0]
    elif isinstance(input_text, dict):
        # If input_text is a dictionary, extract the first value
        input_text = next(iter(input_text.values()), "")

    logger.debug(f"Generating model output for input: {input_text[:100]}{'...' if len(input_text)>100 else ''}")
    try:
        return model.generate(input_text)
    except Exception as e:
        logger.error(f"Error generating model output: {str(e)}")
        return f"Error: {str(e)}"
