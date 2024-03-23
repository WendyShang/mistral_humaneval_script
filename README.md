# mistral_humaneval_script
a simple script to evaluate mistral API human eval models 

* Get mistral API key to access their APIs

* Follow humaneval instructions to set things up [GitHub Pages](https://github.com/openai/human-eval/tree/master).

* generate completion jsonl without async (slow but easier to debug) with `mistral_human_eval.py`

* generate completion json with async (faster but sometimes buggy) with `mistral_human_eval_async.py`

* The post process code primarily comes from: https://github.com/abacaj/code-eval/blob/main/process_eval.py

* mistral small @1 pass rate: 55.5% 

