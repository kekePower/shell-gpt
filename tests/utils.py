from datetime import datetime
from typing import List

import typer
from openai.types.chat import ChatCompletionChunk, ChatCompletionChunkMessage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from typer.testing import CliRunner

from sgpt import main
from sgpt.config import cfg

runner = CliRunner()
app = typer.Typer()
app.command()(main)


def mock_comp(tokens_string: List[str]) -> List[ChatCompletionChunk]:
    """Mock the completion response for testing."""
    chunks = []
    for token in tokens_string:
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model=cfg.get("DEFAULT_MODEL"),
            object="chat.completion.chunk",
            created=int(datetime.now().timestamp()),
            choices=[
                Choice(
                    index=0,
                    finish_reason=None,
                    delta=ChoiceDelta(
                        content=token,
                        role="assistant",
                    ),
                )
            ],
        )
        chunks.append(chunk)
    
    # Add a final chunk with finish_reason="stop"
    if chunks:
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model=cfg.get("DEFAULT_MODEL"),
            object="chat.completion.chunk",
            created=int(datetime.now().timestamp()),
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    delta=ChoiceDelta(),
                )
            ],
        )
        chunks.append(final_chunk)
    
    return chunks


def cmd_args(prompt="", **kwargs):
    arguments = [prompt]
    for key, value in kwargs.items():
        arguments.append(key)
        if isinstance(value, bool):
            continue
        arguments.append(value)
    arguments.append("--no-cache")
    arguments.append("--no-functions")
    return arguments


def comp_args(role, prompt, **kwargs):
    return {
        "messages": [
            {"role": "system", "content": role.role},
            {"role": "user", "content": prompt},
        ],
        "model": cfg.get("DEFAULT_MODEL"),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        **kwargs,
    }
