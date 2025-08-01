from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from typing import List, Union
import concurrent.futures

class BpeTokenizer:
    def __init__(self, path=None):
        self._control_tokens = ["<bos>", "<eos>", "<pad>"]
        self._special_tokens = ["<|user|>", "<|system|>"]
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(behavior="isolated"),
            pre_tokenizers.Metaspace(prepend_scheme="never"),
            pre_tokenizers.Split(pattern=r"(\d+|[a-zA-Z]+|(?:'s|'t|'re|'ve|'m|'ll|'d))", behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.Sequence([
            decoders.ByteLevel(),
            decoders.Metaspace(prepend_scheme="never")
        ])
        tokenizer.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=False)
        ])
        self._tokenizer = tokenizer
        
        if path is not None:
            self._tokenizer = Tokenizer.from_file(path)
    
    def __init_trainer(self, vocab_size, min_freq):
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size  = len(alphabet) + len(self._control_tokens)
        assert vocab_size > min_size
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq, 
            limit_alphabet= vocab_size // 4,
            max_token_length=18,
            special_tokens=self._control_tokens,
            show_progress=True,
            initial_alphabet=alphabet,
        )
        return trainer
        
    def _prepare_trainer_and_tokens(self, vocab_size: int, min_freq: int, reserved_token_size: int) -> tuple:
        assert reserved_token_size > len(self._special_tokens)
        reserved_tokens = [f"<|rsv{i:02d}|>" for i in range(reserved_token_size - len(self._special_tokens))]
        detail_vocab_size = vocab_size - (len(reserved_tokens) + len(self._special_tokens))
        trainer = self.__init_trainer(docab_size=detail_vocab_size, min_freq=min_freq)
        return trainer, detail_vocab_size, reserved_tokens

    def train(self, files, vocab_size, min_freq, reserved_token_size=100):
        trainer, _, reserved_tokens = self._prepare_trainer_and_tokens(
            vocab_size=vocab_size,
            min_freq=min_freq,
            reserved_token_size=reserved_token_size
        )
        self._tokenizer.train(files=files, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens + reserved_tokens)
            
    def train_from_iterator(self, iterator, vocab_size, min_freq, reserved_token_size=100):
        trainer, _, reserved_tokens = self._prepare_trainer_and_tokens(
            vocab_size=vocab_size,
            min_freq=min_freq,
            reserved_token_size=reserved_token_size
        )
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens + reserved_tokens)
            
    def save(self, path):
        self._tokenizer.save(path)
        
    def load(self, path):
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, tokens: Union[str, List[str]], out_ids=True, num_threads=4) -> List:
        if isinstance(tokens, str):
            encoded: Encoding = self._tokenizer.encode(tokens)
            return encoded.ids if out_ids else encoded.tokens
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                encodings: List[Encoding] = list(executor.map(self._tokenizer.encode, tokens))
            
            if out_ids:
                return [encoding.ids for encoding in encodings]
            else:
                return [encoding.tokens for encoding in encodings]

    def decode(self, tokens: List[int], skip_special_tokens=True) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()
    
    @property
    def stop_ids(self) -> List[int]:
        stop_ids = []
        for token in self._control_tokens:
            stop_ids.append(self._tokenizer.token_to_id(token))
        return stop_ids
    
    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id("<bos>")
    
    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id("<eos>")
    
    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id("<pad>")
