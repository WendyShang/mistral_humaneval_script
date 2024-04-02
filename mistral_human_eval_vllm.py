from human_eval.data import write_jsonl, read_problems
from openai import OpenAI

IS_INSTRUCTION = False
num_samples_per_task = 1
temperature = 0.0 # 0.0 is greedy 
#model_name = "/home/wendy/models/mistral-7B-v0.2"
#max_tokens = 32768
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#max_tokens = 8192
#model_name =  "mistralai/Mistral-7B-v0.1" # 31.09% greedy pass@1
#max_tokens = 8192
model_name = "mistralai/Mixtral-8x7B-v0.1" # 43.4% greedy pass@1
max_tokens = 8192

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

problems = read_problems()

def generate_one_completion(prompt):
    if IS_INSTRUCTION:
        prompt_new = f"<s>[INST] WRITE THE FULL FUNCTION TO COMPLETE ```python\n{prompt}```\nEND CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!![/INST]\n### Response:\n```python\n"
        answer = client.completions.create(model=model_name,
                                                temperature = temperature,
                                                max_tokens=max_tokens,
                                                stream = False,
                                                prompt = prompt_new
                                                )        
    else:
        prompt_new = "```python\n" + prompt + "```\n\n### Response:\n```python\n"
        answer = client.completions.create(model=model_name,
                                                temperature = temperature,
                                                max_tokens=max_tokens,
                                                stream = False,
                                                prompt = prompt_new
                                                )
    completion = answer.choices[0].text
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

sub_model_name = model_name.split("/")[-1]
write_jsonl(f"{sub_model_name}_temp_{temperature}_samples_{num_samples_per_task}_pass.jsonl", samples)