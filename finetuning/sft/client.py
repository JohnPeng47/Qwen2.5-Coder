#!/usr/bin/env python3
import requests
import json
import argparse
from typing import Optional, Dict, Any

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate completion using VLLM server
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "aider_models/merged_model/",  # VLLM ignores this
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}\n{response.text}")
            
        return response.json()

def main():
    parser = argparse.ArgumentParser(description='VLLM Client')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to send to server')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='VLLM server URL')
    args = parser.parse_args()

    client = VLLMClient(base_url=args.url)
    
    try:
#         PROMPT = """
# EXCEPTIONS = [
#     ExInfo("APIConnectionError", True, None),
#     ExInfo("APIError", True, None),
#     ExInfo("APIResponseValidationError", True, None),
#     ExInfo(
#         "AuthenticationError",
#         False,
#         "The API provider is not able to authenticate you. Check your API key.",
#     ),
#     ...
# ]

# Fill in the rest of this function from the project Aider, in exceptions.py:
# """
#         prompt = PROMPT
#         response = client.generate(prompt)
#         print("Q: ", prompt)
#         print("A: ", response['choices'][0]['text'])

        TEST_Q = [
            # "Explain the use of word embeddings in Natural Language Processing",
            # "Explain the use of word embeddings in Natural Language Processing",

            # "Explain the use of word embeddings in Natural Language Processing",

            # "Explain the use of word embeddings in Natural Language Processing",

            # "Explain the use of word embeddings in Natural Language Processing",

            # "Explain the use of word embeddings in Natural Language Processing",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",
            # "Can you explain the main functions of the `Analytics` class, particularly regarding how it handles user permissions for collecting and tracking data?",

            # "What strategies does the `Analytics` class employ when encountering errors from tracking services such as Posthog and Mixpanel, and what remedial steps are implemented?",
            # "What factors and conditions does the `Analytics` class evaluate to decide whether analytics should be activated or deactivated?",
            # "What safeguards and privacy protection measures does the `Analytics` class implement when dealing with UUIDs in its data tracking operations?",
            # "In what ways does the Analytics class combine system data and user identifiers to create personalized tracking while respecting user privacy settings?",
            # "Could you explain the main distinctions between the '--gui' and '--browser' command line options in aider's argument parser, and their impact on the graphical interface experience?",
            # "What is the main role and significance of the `get_parser` function found in `aider\\args.py`, and how does it support aider's core functionality?",

            # "What purpose does the '--verify-ssl' argument serve when establishing model connections, and what are the consequences of toggling this setting?",
            # "How does the choice of different models through the '--model' parameter in `aider\\\\args.py` influence the behavior and capabilities of the AI programming assistant?"
        ]
        for q in TEST_Q:
            response = client.generate(q)
            print("Q: ", q)
            print("A: ", response['choices'][0]['text'])
    except Exception as e:
        print(f"Error generating completion: {e}")

if __name__ == "__main__":
    main() 