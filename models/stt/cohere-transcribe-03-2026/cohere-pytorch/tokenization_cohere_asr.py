import os
from typing import Optional

import sentencepiece as spm
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizer
from transformers.utils import cached_file

try:
    from transformers.utils import is_offline_mode
except ImportError:
    from transformers.utils.hub import is_offline_mode
from transformers.utils.import_utils import requires

CMD_ASR_BOS = "<|startoftranscript|>"
CMD_ASR_EOS = "<|endoftext|>"
CMD_ASR_PAD = "<pad>"
CMD_ASR_UNK = "<unk>"
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


@requires(backends=("sentencepiece",))
class CohereAsrTokenizer(PreTrainedTokenizer):
    """
    Cohere ASR tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        spm_model_file: Optional[str] = None,
        bos_token=CMD_ASR_BOS,
        eos_token=CMD_ASR_EOS,
        unk_token=CMD_ASR_UNK,
        pad_token=CMD_ASR_PAD,
        additional_special_tokens=None,
        split_special_tokens=False,
        add_prefix_space=False,
        sp_model_kwargs=None,
        **kwargs,
    ):
        self.spm_model_file = spm_model_file
        self.sp_model_kwargs = sp_model_kwargs or {}
        self.add_prefix_space = add_prefix_space
        self.sp_model = self.get_spm_processor()

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens or [],
            split_special_tokens=split_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.init_kwargs["sp_model_kwargs"] = dict(self.sp_model_kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        local_spm = os.path.join(pretrained_model_name_or_path, "tokenizer.model")
        if os.path.exists(local_spm):
            spm_path = local_spm
        else:
            try:
                spm_path = cached_file(
                    pretrained_model_name_or_path,
                    "tokenizer.model",
                    _raise_exceptions_for_missing_entries=True,
                )
            except EnvironmentError as exc:
                if is_offline_mode():
                    raise ValueError(
                        f"Offline mode: tokenizer.model not found for {pretrained_model_name_or_path}."
                    ) from exc
                raise ValueError(
                    f"tokenizer.model not found in {pretrained_model_name_or_path} (local or remote)."
                ) from exc

        return super().from_pretrained(
            pretrained_model_name_or_path,
            spm_model_file=spm_path,
            *init_inputs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        for token_id, added_token in self.added_tokens_decoder.items():
            if added_token.content not in vocab:
                vocab[added_token.content] = token_id
        return vocab

    def _tokenize(self, text, **kwargs):
        pieces = self.sp_model.encode(text, out_type=str)
        if text and text[0] == " " and (not pieces or pieces[0] != SPIECE_UNDERLINE):
            pieces = [SPIECE_UNDERLINE] + pieces
        return pieces

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.id_to_piece(index)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
            for tok in self.additional_special_tokens or []:
                special_ids.add(self.convert_tokens_to_ids(tok))
            return [1 if tid in special_ids else 0 for tid in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def num_special_tokens_to_add(self, pair=False):
        if pair:
            raise AssertionError(f"Pair sequences not supported for {self.__class__.__name__}.")
        return 2

    def convert_tokens_to_string(self, tokens):
        if not tokens:
            return ""
        if self.add_prefix_space and tokens[0].startswith(SPIECE_UNDERLINE):
            tokens = [tokens[0][1:]] + tokens[1:]
        out = []
        buf = []
        prev_was_special = False

        def flush():
            nonlocal buf, prev_was_special
            if not buf:
                return
            if prev_was_special and buf[0].startswith(SPIECE_UNDERLINE):
                out.append(" ")
            out.append(self.sp_model.decode(buf))
            buf = []
            prev_was_special = False

        for tok in tokens:
            if tok in self.all_special_tokens:
                flush()
                out.append(tok)
                prev_was_special = True
            else:
                buf.append(tok)
        flush()
        return "".join(out)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        out_name = (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model"
        out_path = os.path.join(save_directory, out_name)
        if not os.path.exists(out_path):
            with open(out_path, "wb") as f:
                f.write(self.sp_model.serialized_model_proto())
        return (out_path,)

    def get_spm_processor(self):
        if not self.spm_model_file:
            raise ValueError("CohereAsrTokenizer requires `spm_model_file` (tokenizer.model).")
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        tokenizer.Load(self.spm_model_file)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.sp_model = self.get_spm_processor()
