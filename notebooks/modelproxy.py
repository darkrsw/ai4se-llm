from genutils import CodeGenerator, KeywordsStoppingCriteria, LineBeginStoppingCriteria 
from typing import List
import logging
import torch

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteriaList

class StarCoderProxy(CodeGenerator):
    def __init__(self, ck: str = "bigcode/starcoder", rev: str = "7c6927d25ac2ec0b9e81d98bd54926e36f5c9de1", dev: str = "cuda", stop_words: List[str] = []):
        super().__init__(ck, rev, dev, stop_words)
        self.max_length = 2048
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, revision=self.revision, use_auth_token='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')
        print(f"tokenizer initialized: {self.tokenizer.name_or_path}")

        from constants import load_constants, CONSTANTS
        load_constants("./constants.yaml")

        mem_map = CONSTANTS["DEV_MEM_MAP"]
        print(mem_map)

        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, revision=self.revision, trust_remote_code=True, torch_dtype=torch.float16, device_map="balanced", low_cpu_mem_usage=True, offload_folder="offload", offload_state_dict = True, max_memory=mem_map, use_auth_token='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')
        print(f"model initialized: {self.model.name_or_path} @ {self.model.config._commit_hash}")

        self.tokenizer.pad_token = self.tokenizer.eos_token

        stop_criteria = LineBeginStoppingCriteria(stop_words, self.tokenizer)

        self.stopping_criteria = StoppingCriteriaList([stop_criteria])


class SantaCoderProxy(CodeGenerator):
    def __init__(self, ck: str = "bigcode/santacoder", rev: str = "132eb6b6cedaf579c2f333f1ecd78a16d7e45978", dev: str = "cuda", stop_words: List[str] = []):
        super().__init__(ck, rev, dev, stop_words)
        self.max_length = 2048
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, revision=self.revision, trust_remote_code=True).to(self.device)

        stop_criteria = LineBeginStoppingCriteria(stop_words, self.tokenizer)

        self.stopping_criteria = StoppingCriteriaList([stop_criteria])


    def generate(self, prompt: str):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Adjust the prompt size.
        if inputs.size(dim=1) > (self.max_length/2):
            inputs = inputs[:,-(round(self.max_length/2)):]

        offset = len(inputs.flatten())

        outputs = self.model.generate(inputs, padding_side='left', do_sample=True, temperature=0.5, max_length=self.max_length, stopping_criteria=self.stopping_criteria, pad_token_id=self.tokenizer.eos_token_id)

        return str(self.tokenizer.decode(outputs.flatten()[offset:-1]))
