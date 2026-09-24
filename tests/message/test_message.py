"""The message contract between the API server, the tokenizer worker and the scheduler."""

from __future__ import annotations

import numpy as np

from ftmetal.core import SamplingParams
from ftmetal.message.backend import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from ftmetal.message.frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from ftmetal.message.tokenizer import (
    AbortMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
)


def test_each_message_belongs_to_the_family_its_queue_carries() -> None:
    families = {
        BaseBackendMsg: [BatchBackendMsg, ExitMsg, UserMsg, AbortBackendMsg],
        BaseTokenizerMsg: [
            BatchTokenizerMsg,
            DetokenizeMsg,
            PromptAdmittedMsg,
            TokenizeMsg,
            AbortMsg,
            ErrorReplyMsg,
        ],
        BaseFrontendMsg: [BatchFrontendMsg, UserReply],
    }
    for base, members in families.items():
        assert all(issubclass(member, base) for member in members)
        others = [other for other in families if other is not base]
        assert not any(issubclass(member, other) for member in members for other in others)


def test_messages_have_neutral_defaults() -> None:
    reply = UserReply(uid=1, incremental_output="", finished=False)
    assert (reply.prompt_tokens_delta, reply.completion_tokens_delta) == (0, 0)
    assert reply.error is None and reply.finish_reason is None and reply.matched_stop is None
    first = TokenizeMsg(uid=1, text=[], sampling_params=SamplingParams())
    second = TokenizeMsg(uid=2, text=[], sampling_params=SamplingParams())
    first.chat_template_kwargs["enable_thinking"] = False
    assert second.chat_template_kwargs == {} and first.tools is None
    user = UserMsg(uid=1, input_ids=np.arange(3, dtype=np.int32), sampling_params=SamplingParams())
    batch = BatchBackendMsg(data=[user, AbortBackendMsg(uid=1), ExitMsg()])
    assert [type(msg) for msg in batch.data] == [UserMsg, AbortBackendMsg, ExitMsg]
