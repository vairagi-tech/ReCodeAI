
import os
import re
import subprocess
import tempfile
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json

import google.generativeai as genai
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
# Set your API key as environment variable: GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class ExecutionResult:
    """Stores the result of code execution"""
    success: bool
    output: str
    error: str
    execution_time: float

@dataclass
class PipelineLog:
    """Stores the complete pipeline execution log"""
    prompt: str
    final_code: str
    test_cases: str
    errors_encountered: List[str]
    iteration_count: int
    total_time: float
    success: bool
    timestamp: datetime

class CodeExtractor:
    """Extracts Python code and test cases from LLM responses"""
    
    @staticmethod
    def extract_code_blocks(text: str) -> Dict[str, str]:
        """Extract main code and test code from LLM response"""
        # Pattern to match code blocks
        code_pattern = r'```python\n(.*?)\n```'
        code_blocks = re.findall(code_pattern, text, re.DOTALL)
        
        if len(code_blocks) >= 2:
            return {
                "main_code": code_blocks[0].strip(),
                "test_code": code_blocks[1].strip()
            }
        elif len(code_blocks) == 1:
            # Try to split single block into main and test
            code = code_blocks[0].strip()
            if "def test_" in code or "assert" in code:
                lines = code.split('\n')
                main_lines = []
                test_lines = []
                in_test = False
                
                for line in lines:
                    if line.strip().startswith("def test_") or line.strip().startswith("# Test"):
                        in_test = True
                    
                    if in_test:
                        test_lines.append(line)
                    else:
                        main_lines.append(line)
                
                return {
                    "main_code": '\n'.join(main_lines).strip(),
                    "test_code": '\n'.join(test_lines).strip()
                }
        
        return {"main_code": "", "test_code": ""}

class CodeExecutor:
    """Executes Python code safely in temporary files"""
    
    @staticmethod
    def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
        """Execute Python code and return results"""
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Clean up
            os.unlink(temp_file)
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Code execution timed out",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )

class LLMClient:
    """Handles communication with Gemini API"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_initial_code(self, prompt: str) -> str:
        """Generate initial code and test cases from user prompt"""
        system_prompt = """
You are a Python code generator. Given a task description, generate:
1. A clean, well-documented Python function that solves the task
2. Exactly 3 test cases to validate the function

Format your response as:
```python
# Main function code here
def function_name():
    # Your implementation
    pass
```

```python
# Test cases
def test_function_name():
    # Test case 1
    assert function_name() == expected_result
    
    # Test case 2
    assert function_name() == expected_result
    
    # Test case 3
    assert function_name() == expected_result

# Run tests
if __name__ == "__main__":
    test_function_name()
    print("All tests passed!")
```

Make sure the code is syntactically correct and the tests are meaningful.
"""
        
        try:
            response = self.model.generate_content(f"{system_prompt}\n\nTask: {prompt}")
            return response.text
        except Exception as e:
            logger.error(f"Error generating initial code: {e}")
            return ""
    
    def fix_code(self, original_code: str, error_message: str, iteration: int) -> str:
        """Ask LLM to fix code based on error message"""
        fix_prompt = f"""
The following Python code has an error:

```python
{original_code}
```

Error message: {error_message}

This is iteration {iteration} of fixing. Please provide the corrected code in the same format as before:
- First code block: Main function code
- Second code block: Test cases

Make sure to fix the specific error and maintain the same functionality.
"""
        
        try:
            response = self.model.generate_content(fix_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error fixing code: {e}")
            return ""

class SelfHealingPipeline:
    """Main pipeline that orchestrates the self-healing process"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.code_extractor = CodeExtractor()
        self.code_executor = CodeExecutor()
        self.max_iterations = 10
    
    def run_pipeline(self, prompt: str) -> PipelineLog:
        """Run the complete self-healing pipeline"""
        start_time = time.time()
        errors_encountered = []
        iteration_count = 0
        
        logger.info(f"Starting pipeline for prompt: {prompt}")
        
        # Generate initial code
        llm_response = self.llm_client.generate_initial_code(prompt)
        if not llm_response:
            return self._create_failed_log(prompt, "Failed to generate initial code", start_time)
        
        code_blocks = self.code_extractor.extract_code_blocks(llm_response)
        main_code = code_blocks.get("main_code", "")
        test_code = code_blocks.get("test_code", "")
        
        if not main_code or not test_code:
            return self._create_failed_log(prompt, "Failed to extract code blocks", start_time)
        
        # Self-healing loop
        for iteration in range(self.max_iterations):
            iteration_count = iteration + 1
            logger.info(f"Iteration {iteration_count}: Testing code")
            
            # Test main code
            main_result = self.code_executor.execute_code(main_code)
            if not main_result.success:
                error_msg = f"Main code error (iteration {iteration_count}): {main_result.error}"
                errors_encountered.append(error_msg)
                logger.warning(error_msg)
                
                # Ask LLM to fix the code
                fix_response = self.llm_client.fix_code(main_code, main_result.error, iteration_count)
                if not fix_response:
                    break
                
                new_code_blocks = self.code_extractor.extract_code_blocks(fix_response)
                main_code = new_code_blocks.get("main_code", main_code)
                test_code = new_code_blocks.get("test_code", test_code)
                continue
            
            # Test the test cases
            test_result = self.code_executor.execute_code(test_code)
            if not test_result.success:
                error_msg = f"Test code error (iteration {iteration_count}): {test_result.error}"
                errors_encountered.append(error_msg)
                logger.warning(error_msg)
                
                # Ask LLM to fix the tests
                combined_code = f"{main_code}\n\n{test_code}"
                fix_response = self.llm_client.fix_code(combined_code, test_result.error, iteration_count)
                if not fix_response:
                    break
                
                new_code_blocks = self.code_extractor.extract_code_blocks(fix_response)
                main_code = new_code_blocks.get("main_code", main_code)
                test_code = new_code_blocks.get("test_code", test_code)
                continue
            
            # Success!
            logger.info(f"Pipeline completed successfully after {iteration_count} iterations")
            return PipelineLog(
                prompt=prompt,
                final_code=main_code,
                test_cases=test_code,
                errors_encountered=errors_encountered,
                iteration_count=iteration_count,
                total_time=time.time() - start_time,
                success=True,
                timestamp=datetime.now()
            )
        
        # Max iterations reached
        final_error = f"Maximum iterations ({self.max_iterations}) reached without success"
        errors_encountered.append(final_error)
        
        return PipelineLog(
            prompt=prompt,
            final_code=main_code,
            test_cases=test_code,
            errors_encountered=errors_encountered,
            iteration_count=iteration_count,
            total_time=time.time() - start_time,
            success=False,
            timestamp=datetime.now()
        )
    
    def _create_failed_log(self, prompt: str, error: str, start_time: float) -> PipelineLog:
        """Create a failed pipeline log"""
        return PipelineLog(
            prompt=prompt,
            final_code="",
            test_cases="",
            errors_encountered=[error],
            iteration_count=0,
            total_time=time.time() - start_time,
            success=False,
            timestamp=datetime.now()
        )

# FastAPI Application
app = FastAPI(title="Self-Healing LLM Pipeline", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Global pipeline instance
pipeline = SelfHealingPipeline()

# In-memory storage for logs (in production, use a database)
execution_logs: List[PipelineLog] = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with prompt input form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_code(request: Request, prompt: str = Form(...)):
    """Generate and self-heal code based on user prompt"""
    if not prompt.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please enter a valid prompt"
        })
    
    # Run the pipeline
    log = pipeline.run_pipeline(prompt)
    
    # Store the log
    execution_logs.append(log)
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "log": log
    })

@app.get("/logs", response_class=HTMLResponse)
async def view_logs(request: Request):
    """View all execution logs"""
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": execution_logs[-10:]  # Show last 10 logs
    })

@app.get("/api/logs")
async def get_logs_api():
    """API endpoint to get logs as JSON"""
    return [asdict(log) for log in execution_logs[-10:]]

if __name__ == "__main__":
    import uvicorn
    
    # Create templates directory and files
    os.makedirs("templates", exist_ok=True)
    
   
    
    print(" Starting Self-Healing LLM Pipeline server...")
    print(" Make sure to set your GEMINI_API_KEY environment variable!")
    print(" Open http://localhost:8000 in your browser")

    uvicorn.run(app, host="0.0.0.0", port=8000)