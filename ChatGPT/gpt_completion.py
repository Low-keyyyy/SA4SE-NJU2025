import openai
import time
import timeout_decorator


openai.api_key  = 'YOUR_KEY'

turbo_name = "gpt-3.5-turbo-0613"
big_turbo_name = "gpt-3.5-turbo-16k-0613"

def get_completion_by_loop(get_completion,prompt):
    while True:
        try:
            response = get_completion(prompt)
            # Check if the response has a value
            if response is not None:
                break  # If the response has a value, jump out of the loop
        except timeout_decorator.TimeoutError as e:
            print(f"Timeout exception: {e}, Retrying currently...")
        except Exception as e:
            sec = 30
            print(f"An exception has occurred: {e}, Retrying in {sec} seconds...")
            time.sleep(sec)  # Wait and rerun the code

    return response

@timeout_decorator.timeout(90)
def get_turbo_completion_with_prompt(prompt, model=turbo_name):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

@timeout_decorator.timeout(90)
def get_turbo_completion_with_messages(messages, model=turbo_name):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

@timeout_decorator.timeout(90)
def get_big_turbo_completion_with_prompt(prompt, model=big_turbo_name):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

@timeout_decorator.timeout(90)
def get_big_turbo_completion_with_messages(messages, model=big_turbo_name):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_completion_from_pmt_with_turbo(prompt):
    response = get_completion_by_loop(get_turbo_completion_with_prompt, prompt)
    return response

def get_completion_from_msg_with_turbo(messages):
    response = get_completion_by_loop(get_turbo_completion_with_messages, messages)
    return response

def get_completion_from_pmt_with_big_turbo(prompt):
    response = get_completion_by_loop(get_big_turbo_completion_with_prompt, prompt)
    return response

def get_completion_from_msg_with_big_turbo(messages):
    response = get_completion_by_loop(get_big_turbo_completion_with_messages, messages)
    return response


