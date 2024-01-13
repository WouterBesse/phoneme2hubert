from typing import List, Iterable, Dict, Tuple, Any

class SequenceTokenizer:

    """Tokenizes text and optionally attaches language-specific start index (and non-specific end index)."""

    def __init__(self,
                 symbols: List[str],
                 languages: List[str],
                 char_repeats: int,
                 lowercase: bool = True,
                 append_start_end: bool =True,
                 pad_token='_',
                 end_token='<end>') -> None:
        """
        Initializes a SequenceTokenizer object.

        Args:
          symbols (List[str]): Character (or phoneme) symbols.
          languages (List[str]): List of languages.
          char_repeats (int): Number of repeats for each character to allow the forward model to map to longer
               phoneme sequences. Example: for char_repeats=2 the tokenizer maps hi -> hhii.
          lowercase (bool): Whether to lowercase the input word.
          append_start_end (bool): Whether to append special start and end tokens. Start and end tokens are
               index mappings of the chosen language.
          pad_token (str): Special pad token for index 0.
          end_token (str): Special end of sequence token.
        """

        
        self.languages = languages
        self.lowercase = lowercase
        self.char_repeats = char_repeats
        self.append_start_end = append_start_end
        self.pad_index = 0
        self.token_to_idx = {pad_token: self.pad_index}
        self.special_tokens = {pad_token, end_token}
        for lang in languages:
            lang_token = self._make_start_token(lang)
            self.token_to_idx[lang_token] = len(self.token_to_idx)
            self.special_tokens.add(lang_token)
        self.token_to_idx[end_token] = len(self.token_to_idx)
        self.end_index = self.token_to_idx[end_token]
        for symbol in symbols:
            self.token_to_idx[symbol] = len(self.token_to_idx)
        self.idx_to_token = {i: s for s, i in self.token_to_idx.items()}
        self.vocab_size = len(self.idx_to_token)

    def __call__(self, sentence: Iterable[str], language: str) -> List[int]:
        """
        Maps a sequence of symbols for a language to a sequence of indices.

        Args:
          sentence  (Iterable[str]): Sentence (or word) as a sequence of symbols.
          language (str): Language for the mapping that defines the start and end token indices.

        Returns:
           List[int]: Sequence of token indices.
        """

        sentence = [item for item in sentence for i in range(self.char_repeats)]
        if language not in self.languages:
            raise ValueError(f'Language not supported: {language}. Supported languages: {self.languages}')
        if self.lowercase:
            sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if self.append_start_end:
            sequence = [self._get_start_index(language)] + sequence + [self.end_index]
        return sequence

    def decode(self, sequence: Iterable[int], remove_special_tokens: bool = False) -> List[str]:
        """Maps a sequence of indices to a sequence of symbols.

        Args:
          sequence (Iterable[int]): Encoded sequence to be decoded.
          remove_special_tokens (bool): Whether to remove special tokens such as pad or start and end tokens. (Default value = False)
          sequence: Iterable[int]: 

        Returns:
           List[str]: Decoded sequence of symbols.
        """

        sequence = list(sequence)
        if self.append_start_end:
            sequence = sequence[:1] + sequence[1:-1:self.char_repeats] + sequence[-1:]
        else:
            sequence = sequence[::self.char_repeats]
        decoded = [self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token]
        if remove_special_tokens:
            decoded = [d for d in decoded if d not in self.special_tokens]
        return decoded

    def _get_start_index(self, language: str) -> int:
        lang_token = self._make_start_token(language)
        return self.token_to_idx[lang_token]

    def _make_start_token(self, language: str) -> str:
        return '<' + language + '>'