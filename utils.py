import json
from typing import Any, Dict, Iterator, List, Literal, Optional, Union
import jinja2
import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar

def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {}
                    ),
                    "logprobs": chunk["choices"][0]["logprobs"],
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }

def _convert_text_completion_to_chat(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "logprobs": completion["choices"][0]["logprobs"],
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }

def _grammar_for_json(verbose:bool=False):
    return llama_grammar.LlamaGrammar.from_string(llama_grammar.JSON_GBNF, verbose=verbose)

def _grammar_for_json_schema(
        schema: str,
        verbose: bool = False,
        fallback_to_json: bool = True
):
    try:
        return llama_grammar.LlamaGrammar.from_json_schema(schema, verbose=verbose)
    except Exception as e:
        if fallback_to_json:
            return _grammar_for_json(verbose=verbose)
        else:
            raise e

def _grammar_for_response_format(
        response_format: llama_types.ChatCompletionRequestResponseFormat,
        verbose: bool = False
):
    if response_format["type"] != "json_object":
        return None

    if "schema" in response_format:
        return _grammar_for_json_schema(
            json.dumps(response_format["schema"]), verbose=verbose
        )
    else:
        return _grammar_for_json(verbose=verbose)

def _convert_completion_to_chat_function(
    tool_name: str,
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool,
):
    if not stream:
        completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
        assert "usage" in completion
        tool_id = "call_" + "_0_" + tool_name + "_" + completion["id"]
        # TODO: Fix for legacy function calls
        chat_completion: llama_types.CreateChatCompletionResponse = {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": tool_name,
                            "arguments": completion["choices"][0]["text"],
                        },
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion["choices"][0]["text"],
                                },
                            }
                        ],
                    },
                    "logprobs": completion["choices"][0]["logprobs"],
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": completion["usage"],
        }
        return chat_completion
    else:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore

        def _stream_response_to_function_stream(
            chunks: Iterator[llama_types.CreateCompletionStreamResponse],
        ) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
            # blank first message
            first = True
            id_ = None
            created = None
            model = None
            tool_id = None
            for chunk in chunks:
                if first:
                    id_ = "chat" + chunk["id"]
                    created = chunk["created"]
                    model = chunk["model"]
                    tool_id = "call_" + "_0_" + tool_name + "_" + chunk["id"]
                    yield {
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": None,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "function_call": None,
                                    "tool_calls": None,
                                },
                            }
                        ],
                    }
                    yield {
                        "id": "chat" + chunk["id"],
                        "object": "chat.completion.chunk",
                        "created": chunk["created"],
                        "model": chunk["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": chunk["choices"][0]["logprobs"],
                                "delta": {
                                    "role": None,
                                    "content": None,
                                    "function_call": {
                                        "name": tool_name,
                                        "arguments": chunk["choices"][0]["text"],
                                    },
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": chunk["choices"][0]["text"],
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                    first = False
                    continue
                assert tool_id is not None
                yield {
                    "id": "chat" + chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": chunk["choices"][0]["logprobs"],
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": {
                                    "name": tool_name,
                                    "arguments": chunk["choices"][0]["text"],
                                },
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": chunk["choices"][0][
                                                "text"
                                            ],
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }

            if id_ is not None and created is not None and model is not None:
                yield {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "logprobs": None,
                            "delta": {
                                "role": None,
                                "content": None,
                                "function_call": None,
                                "tool_calls": None,
                            },
                        }
                    ],
                }

        return _stream_response_to_function_stream(chunks)
    
def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)
