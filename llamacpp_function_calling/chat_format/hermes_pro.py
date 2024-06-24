import json
from typing import (
    Any, 
    Dict, 
    Iterator, 
    List, 
    Literal, 
    Optional, 
    Union
)

from ..utils import (
    _grammar_for_response_format,
    _convert_completion_to_chat,
    _convert_completion_to_chat_function
) 

import jinja2
import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar


def hermes_pro_function_calling(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:  
    
    function_calling_template = (
        "<|begin_of_text|>"
        "{% if tool_calls %}"
        "<|im_start|>system\n"
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:"
        "<tools>\n"
        "{% for tool in tools %}"
        "{{ tool.function | tojson }}\n"
        "{% endfor %}"
        "</tools>"
        """Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}"""
        "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:"
        "\n<tool_call>"
        """\n{"arguments": <args-dict>, "name": <function-name>}"""
        "\n</tool_call><|im_end|>"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message.role == 'system' or message.role == 'assistant'%}"
        "\n<|im_start|>{{ message.role }}"
        "\n{{ message.content }}<|im_end|>"
        "{% elif message.role == 'tool' %}"
        "\n<|im_start|>{{ message.role }}"
        "\n<tool_response>"
        "\n{{ message.content }}"
        "\n</tool_response>"
        "\n<|im_end|>"
        "{% else %}"
        "\n<|im_start|>{{ message.role }}"
        "\n{{ message.content }}<|im_end|>"
        "{% endif %}"
        "{% endfor %}"
        "\n<|im_start|>assistant"
    )

    template_renderer = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        undefined=jinja2.StrictUndefined,
    ).from_string(function_calling_template)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = [stop, "<|im_end|>"] if isinstance(stop, str) else stop + ["<|im_end|>"] if stop else ["<|im_end|>"]

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2: Tool choice by user
    if isinstance(tool_choice, dict):
        tool_name = tool_choice["function"]["name"]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == tool_name), None
        )
        if tool is None:
            raise ValueError(f"Tool with name '{tool_name}' not found in tools")
        prompt = template_renderer.render(
            messages=messages,
            tools=tools,
            tool_calls=True,
            add_generation_prompt=True,
        )
        prompt += f"functions.{tool_name}:\n"
        try:
            grammar = llama_grammar.LlamaGrammar.from_json_schema(
                json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
            )
        except Exception as e:
            grammar = llama_grammar.LlamaGrammar.from_string(
                llama_grammar.JSON_GBNF, verbose=llama.verbose
            )
            if llama.verbose:
                print(
                    "Failed to parse function body as JSON schema, falling back to default grammar"
                )
                print(e)
        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        return _convert_completion_to_chat_function(
            tool_name, completion_or_chunks, stream
        )

    # Case 3: Automatic tool choice
    assert isinstance(tool_choice, str) and tool_choice == "auto"
    
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )

    # print('Prompt_template:\n',prompt)
    
    completion_or_chunks = llama.create_completion(
        prompt=prompt,
        temperature=0,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
    )
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    text = completion["choices"][0]["text"]

    # Not calling a tool, return raw messages
    if "<tool_call>" not in text:
        return _convert_completion_to_chat(completion)
    
    
    # Extract the JSON string from the response
    json_string = text[text.find("{"):text.rfind("}") + 1]

    # Convert JSON string to dictionary
    tool_call_dict = json.loads(json_string)

    stream=False

    if not stream:

        # Merge completions
        function_call_dict: Union[Dict[str, str], Dict[Literal["function_call"], llama_types.ChatCompletionRequestAssistantMessageFunctionCall]] = { 
            "function_call": {
                "name": tool_name,
                "arguments": completion[0]["choices"][0]["text"],
            }
        } if len(completion) == 1 else {}
        
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": completion["choices"][0]["logprobs"],
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_"
                                + f"_{0}_"
                                + tool_call_dict["name"]
                                + "_"
                                + completion["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call_dict["name"],
                                    "arguments": tool_call_dict["arguments"],
                                },
                            }
                            # for i, (tool_name, completion) in enumerate(
                            #     zip(completions_tool_name, completions)
                            # )
                        ],
                        **function_call_dict
                    },
                }
            ],
            "usage": {
                "completion_tokens": 
                    completion["usage"]["completion_tokens"] if "usage" in completion else 0
                ,
                "prompt_tokens": 
                    completion["usage"]["prompt_tokens"] if "usage" in completion else 0
                ,
                "total_tokens": 
                    completion["usage"]["total_tokens"] if "usage" in completion else 0
                ,
            },
        }

    raise ValueError("Automatic streaming tool choice is not supported")
