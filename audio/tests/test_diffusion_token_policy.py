import pytest
from types import SimpleNamespace

from audio.diffusion.token_policy import (
    build_diffusion_token_policy,
    hash_token_ids,
    validate_mawjais_token_contract,
)


class TinyTokenizer:
    vocab_size = 8

    def __len__(self):
        return 12

    def get_added_vocab(self):
        # BOS/PAD can appear in this mapping even though their IDs are below
        # tokenizer.vocab_size, exactly as in the MawJais tokenizer.
        return {"<bos>": 0, "<pad>": 1, "<turn_end>": 9, "<audio>": 10, "<mask>": 11}


def build(scope="safe_vocab", tail="eos_repeat"):
    return build_diffusion_token_policy(
        TinyTokenizer(),
        model_vocab_size=12,
        uniform_vocab_scope=scope,
        allowed_added_output_token_ids=(9,),
        extra_forbidden_output_token_ids=(0, 1, 10, 11),
        mask_token_id=11,
        tail_fill_mode=tail,
        pad_token_id=1,
        turn_end_token_id=9,
    )


def test_safe_prior_excludes_controls_and_keeps_turn_end():
    policy = build()
    assert policy.uniform_noise_ids == (2, 3, 4, 5, 6, 7, 9)
    assert set(policy.sampling_forbidden_ids) == {0, 1, 10, 11}
    assert policy.uniform_noise_ids_sha256 == hash_token_ids(policy.uniform_noise_ids)


def test_full_prior_really_uses_every_model_row():
    policy = build("full_vocab")
    assert policy.uniform_noise_ids == tuple(range(12))
    assert 10 in policy.sampling_forbidden_ids


def test_semantic_pad_is_allowed_only_for_eos_then_pad():
    assert 1 in build(tail="eos_repeat").training_forbidden_ids
    assert 1 not in build(tail="eos_then_pad").training_forbidden_ids


def test_turn_end_cannot_be_removed_from_allowed_added_outputs():
    with pytest.raises(ValueError, match="turn-end"):
        build_diffusion_token_policy(
            TinyTokenizer(),
            model_vocab_size=12,
            uniform_vocab_scope="safe_vocab",
            allowed_added_output_token_ids=(),
            extra_forbidden_output_token_ids=(),
            mask_token_id=11,
            tail_fill_mode="eos_repeat",
            pad_token_id=1,
            turn_end_token_id=9,
        )


class ContractTokenizer:
    pad_token_id = 1

    def convert_tokens_to_ids(self, token):
        return {
            "reserved_special_token_14": 14,
            "<|ifm|im_end|>": 19,
            "<|ifm|audio|>": 62,
            "<audio>": 35,
            "</audio>": 36,
        }.get(token, 0)


def _validate_contract(**overrides):
    arguments = {
        "processor": SimpleNamespace(audio_token_id=62),
        "model_config": SimpleNamespace(
            audio_token_id=62,
            audio_start_token_id=35,
            audio_end_token_id=36,
        ),
        "mask_token_text": "reserved_special_token_14",
        "mask_token_id": 14,
        "pad_token_id": 1,
        "turn_end_token_id": 19,
        "forbidden_output_token_ids": (14, 35, 36, 62),
    }
    arguments.update(overrides)
    return validate_mawjais_token_contract(ContractTokenizer(), **arguments)


def test_mawjais_control_contract_cross_checks_tokenizer_processor_and_model():
    contract = _validate_contract()
    assert contract.audio_token_id == 62
    assert contract.turn_end_token_id == 19
    assert contract.to_metadata()["mask_token_text"] == "reserved_special_token_14"

    with pytest.raises(ValueError, match="processor.audio_token_id"):
        _validate_contract(processor=SimpleNamespace(audio_token_id=99))
    with pytest.raises(ValueError, match="audio_start_token_id"):
        _validate_contract(
            model_config=SimpleNamespace(
                audio_token_id=62,
                audio_start_token_id=99,
                audio_end_token_id=36,
            )
        )
    with pytest.raises(ValueError, match="does not forbid"):
        _validate_contract(forbidden_output_token_ids=(14, 62))
