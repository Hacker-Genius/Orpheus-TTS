import asyncio
import torch
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
import threading
import queue
from .decoder import tokens_decoder_sync


class OrpheusModel:
    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        tokenizer='canopylabs/orpheus-3b-0.1-pretrained',
        hf_token=None,
        **engine_kwargs
    ):
        # Authenticate with Hugging Face if token is provided
        if hf_token:
            login(token=hf_token)
        elif os.getenv('HF_TOKEN'):
            login(token=os.getenv('HF_TOKEN'))
            
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs  # vLLM engine kwargs
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
        
        # Use provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
    
    def _map_model_params(self, model_name):
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(
                f"Model {model_name} is not supported. Only medium-3b is supported, "
                "small, micro and nano models will be released very soon"
            )
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name
        
    def _setup_engine(self):
        # Get token from environment or kwargs
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            self.engine_kwargs['token'] = hf_token
            
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            **self.engine_kwargs
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def validate_voice(self, voice):
        if voice and voice not in self.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def generate_tokens_sync(
        self,
        prompt,
        voice=None,
        request_id="req-001",
        temperature=0.6,
        top_p=0.8,
        max_tokens=1200,
        stop_token_ids=[49158],
        repetition_penalty=1.3
    ):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(
                prompt=prompt_string,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()
    
    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs)) 