import os
from typing import List

from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteriaList, MaxLengthCriteria, StoppingCriteria
import torch


class CodeGenerator:
    def __init__(self, ck: str, rev: str, dev: str, stop_words: List[str]):
        self.checkpoint = ck
        self.revision = rev
        self.device = dev # either of "cpu", "cuda", or "mps".
        self.stop_words = stop_words

    def generate(prompt: str):
        pass # to be overriden.



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list, tnizer):
        self.keywords = keywords_ids
        self.tokenizer = tnizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # print(self.tokenizer.decode(input_ids[0][-1]), input_ids[0][-1], self.keywords, input_ids[0][-1] in self.keywords)
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class LineBeginStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords:list, tnizer):
        self.keywords = keywords
        self.tokenizer = tnizer
        self.keyids = []

        vocabmap = self.tokenizer.vocab

        for bpe, tid in vocabmap.items():
            decoded = self.tokenizer.decoder.decode([bpe])
            for keyword in self.keywords:
                if decoded.strip().startswith(keyword):
                    self.keyids.append(tid)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keyids:
            return True
        return False
