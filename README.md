# llm-president-quiz

```bash
pip install -r requirements.txt
python llm_survey.py
python llm_survey.py --provider gemini --model gemini-3-pro-preview \
    -q "Who is the US President?" \  # input the question to ask
    -n 10 \  # how many surveys to take
python tally.py -i answers_20251106_220928.csv --categories "Joe Biden", "Donald Trump"
```

Environment variables:

* `OPENAI_API_KEY` (required for OpenAI mode)
* `GEMINI_API_KEY` or `GOOGLE_API_KEY` (required for Gemini mode)
