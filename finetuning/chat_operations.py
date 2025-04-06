# chat_operations.py

import json  # For handling JSON data
import logging  # For logging messages
import os  # For interacting with the operating system (e.g., environment variables)

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MV_OPENAI_KEY"]
)  # OpenAI API for interacting with language models

# System message that sets the context for the assistant
SYSTEM_MESSAGE = (
    "You are an information extraction and retrieval agent. "
    "Your search is to run functions on and answer questions about a piece of text that I provide you. "
    "Answer in as few tokens as possible. The end-users have access to the source material and are educated scientists. "
    "You do not have to explain terms or definition. If you do not know an answer to a question, only answer 'unreported' with no additional text. "
    "Use expert terminology and quantify where possible. Only report the requested information/only answer the specified question. Answer in as few tokens as possible."
)


def sessionless_message(system_message, message, model):
    """
    Sends a message to the OpenAI API without keeping track of the conversation history.

    Args:
        system_message (str): The system message that sets the context for the assistant.
        message (str): The message to send to the assistant.
        model (str): The model to use for the conversation.

    Returns:
        assistant_reply (str): The assistant's reply.
        output_tokens (int): The number of output tokens used.
        input_tokens (int): The number of input tokens used.
        model (str): The model used for the conversation.
    """
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": message,
        },
    ]

    # Send the conversation history to the OpenAI API and get the assistant's response
    response = client.chat.completions.create(
        model=model,  # Specify the model to use
        messages=messages,  # Provide the conversation history
        timeout=180,
    )

    # Extract the assistant's reply from the response
    assistant_reply = response.choices[0].message.content

    # Retrieve token usage information from the response
    output_tokens = response.usage.completion_tokens
    input_tokens = response.usage.prompt_tokens

    # Return the assistant's reply and token usage details
    return assistant_reply


def sessionless_call_function(system_message, message, schema, model):
    """
    Sends a message to the OpenAI API without keeping track of the conversation history.

    Args:
        system_message (str): The system message that sets the context for the assistant.
        message (str): The message to send to the assistant.
        schema (): The desired schema of the structured JSON output.
        model (str): The model to use for the conversation.

    Returns:
        assistant_reply (str): The assistant's reply.
        output_tokens (int): The number of output tokens used.
        input_tokens (int): The number of input tokens used.
    """

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": message,
        },
    ]

    # Make LLM function call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=120,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "json_schema",
                "strict": True,
                "schema": schema,
            },
        },
    )

    assistant_reply = response.choices[0].message.content

    output_tokens = response.usage.completion_tokens
    input_tokens = response.usage.prompt_tokens

    return assistant_reply


def sessionless_vision_call_function(system_message, text_message, base64_image, image_media_type, schema, model):
    """
    Sends a text message and a base64 encoded image to the OpenAI API 
    using a vision-capable model, without keeping track of conversation history.

    Args:
        system_message (str): The system message setting the context.
        text_message (str): The text part of the user message.
        base64_image (str): The base64 encoded string of the image.
        image_media_type (str): The media type of the image (e.g., "image/png", "image/jpeg").
        schema (dict): The desired JSON schema for the output.
        model (str): The vision-capable model to use (e.g., "gpt-4-vision-preview", "gpt-4o").

    Returns:
        assistant_reply (str): The assistant's structured JSON reply.
        # Note: Token usage might be reported differently or not at all for vision models depending on API version/model
    """

    messages = [
        {
            "role": "system", 
            "content": system_message
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": text_message
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{base64_image}"
                    },
                },
            ],
        },
    ]

    # Make LLM function call
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=120,
            response_format={
                "type": "json_schema", # Vision models often use json_object type
                # Schema definition might vary slightly depending on the exact API usage for vision models
                # This assumes a compatible setup. Check OpenAI docs if issues arise.
                 "json_schema": {
                     "name": "json_schema",
                     "strict": True,
                     "schema": schema,
                 },
             },
            # max_tokens might be useful here depending on expected output size
            # max_tokens=300 
        )
        assistant_reply = response.choices[0].message.content
        # TODO: Add token counting if available and needed for vision model
        return assistant_reply
    except Exception as e:
        print(f"Error calling OpenAI Vision API: {e}")
        return None # Indicate failure