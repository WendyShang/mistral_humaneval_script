from human_eval.data import write_jsonl, read_problems
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
os.environ["MISTRAL_API_KEY"] = # Your API key here
mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
model_name = "mistral-small-latest"
num_samples_per_task = 1

problems = read_problems()

def generate_one_completion(prompt):
    prompt_new = "```python\n" + prompt + "```\n\n### Response:\n```python\n"
    answer = mistral_client.chat(model=model_name,
                                         temperature = 0.5,
                                           messages= [
                                               ChatMessage(role="user", content=prompt_new)
                                               ],
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

task_ids = list(problems.keys())
samples = []
for i in range(len(task_ids)):
    task_id = task_ids[i]
    print(f"Generating completion for task {i+1}/{len(task_ids)}: {task_id}")
    count = 0 
    while True:
        try:
            current = [
                dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
                for _ in range(num_samples_per_task)
            ]
            samples.extend(current)
            break
        except Exception as e:
            count += 1
            if count >= 3:
                raise e

write_jsonl(f"{model_name}_samples_{num_samples_per_task}_pass.jsonl", samples)