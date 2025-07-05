# ReCodeAi: Self-Healing LLM Pipeline

ReCodeAi is an AI-powered, self-healing Python code generation pipeline. It leverages Google Gemini LLM to generate code and test cases from a user prompt, automatically executes and tests the code, and iteratively fixes errors until the code works or a maximum number of iterations is reached. The web app provides a simple interface to interact with the pipeline and view execution logs.

---

## Features
- **Prompt-to-Code**: Enter a Python coding task, and the AI generates the function and test cases.
- **Self-Healing**: Automatically detects errors, asks the LLM to fix them, and retries until success or max iterations (default: 10).
- **Test Automation**: Runs generated test cases for every iteration.
- **Full Transparency**: View errors encountered, fixes applied, and logs of all recent runs.
- **Modern Web UI**: Clean, responsive interface using FastAPI and Jinja2 templates.

---

## How It Works
1. **User Input**: Enter a description of the Python function you want.
2. **Code Generation**: Gemini LLM generates both the function and three test cases.
3. **Execution & Testing**: The pipeline executes the code and runs the tests.
4. **Error Handling**: If errors occur, the error message is sent back to the LLM for an improved version.
5. **Iteration**: Steps 3-4 repeat until the code passes all tests or the maximum number of iterations is reached.
6. **Logs**: All attempts, errors, and fixes are logged and viewable in the UI.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/vairagi-tech/ReCodeAI.git
cd ReCodeAi
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up Gemini API Key
Export your [Google Gemini API key](https://ai.google.dev/) as an environment variable:
```bash
export GEMINI_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
python app.py
```

The app will be available at [http://localhost:8000](http://localhost:8000).

---

## Project Structure
```
├── app.py               # Main FastAPI application and pipeline logic
├── templates/           # Jinja2 HTML templates (UI)
│   ├── index.html       # Home page (prompt input)
│   ├── result.html      # Generation result page
│   └── logs.html        # Execution logs page
├── .venv/               # (Optional) Python virtual environment
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Requirements
- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Jinja2](https://palletsprojects.com/p/jinja/)
- [google-generativeai](https://pypi.org/project/google-generativeai/)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Environment Variables
- `GEMINI_API_KEY` (**required**): Your Google Gemini API key for LLM access.

---

## Usage
1. Go to http://localhost:8000
2. Enter a Python coding task (e.g., "Create a function that returns the nth Fibonacci number").
3. View generated code, test cases, and the self-healing process in real time.
4. Review the logs for previous executions and error fixes.

---

## License
MIT License

---

## Acknowledgements
- [Google Gemini](https://ai.google.dev/) for LLM code generation
- [FastAPI](https://fastapi.tiangolo.com/) for web framework
- [Jinja2](https://palletsprojects.com/p/jinja/) for templating

---

## Troubleshooting
- **GEMINI_API_KEY not set**: Ensure you have exported your Gemini API key before running the app.
- **Code execution errors**: Errors in generated code are automatically fixed by the pipeline. If persistent, review logs for details.

---

For questions or issues, please open an issue on the repository.
