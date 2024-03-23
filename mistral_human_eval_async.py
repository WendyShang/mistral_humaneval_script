import asyncio
from human_eval.data import write_jsonl, read_problems
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from concurrent.futures import ThreadPoolExecutor

concurrent_calls = 10  
semaphore = asyncio.Semaphore(concurrent_calls)

os.environ["MISTRAL_API_KEY"] = # Your API key here
mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
model_name = "mistral-small-latest"
num_samples_per_task = 1

problems = read_problems()

def generate_one_completion(prompt):
    prompt_new = "```python\n" + prompt + "```\n\n### Response:\n```python\n"
    answer = mistral_client.chat(model=model_name,
                                 temperature=0.5,
                                 messages=[ChatMessage(role="user", content=prompt_new)],
                                 )
    completion = answer.choices[0].message.content
    completion = completion.replace("\r", "")
    if "```python" in completion:
        def_line = completion.index("```python")
        completion = completion[def_line:].strip()
        completion = completion.replace("```python", "")
        try:
            next_line = completion.index("```")
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")
    if '__name__ == "__main__"' in completion:
        next_line = completion.index('if __name__ == "__main__":')
        completion = completion[:next_line].strip()

    if "# Example usage" in completion:
        next_line = completion.index("# Example usage")
        completion = completion[:next_line].strip()
    try:
        next_line = completion.index("```")
        completion = completion[:next_line]
    except:
        pass
    return completion

async def generate_one_completion_async(prompt):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        async with semaphore:  
            completion = await loop.run_in_executor(pool, generate_one_completion, prompt)
    return completion

async def generate_completions_for_task(task_id, prompt):
    print("generating for task", task_id)
    completions = []
    for _ in range(num_samples_per_task):
        try:
            completion = await generate_one_completion_async(prompt)
            completions.append(dict(task_id=task_id, completion=completion))
        except Exception as e:
            print(f"Error generating completion for task {task_id}: {e}")
    return completions

async def main():
    task_ids = list(problems.keys())
    samples = []

    tasks = [generate_completions_for_task(task_id, problems[task_id]["prompt"]) for task_id in task_ids]
    results = await asyncio.gather(*tasks)

    for result in results:
        samples.extend(result)

    write_jsonl(f"{model_name}_samples_{num_samples_per_task}_pass.jsonl", samples, append=True)


asyncio.run(main())