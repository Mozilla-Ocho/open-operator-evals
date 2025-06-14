import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import override

from eval.data.load_data import BenchmarkTask
from eval.task_types import AgentBenchmark, Step, TaskResult, LLMCall

class SparkInput(BaseModel):
    spark_dir: str = "../spark"
    env: Optional[Dict[str, str]] = None
    timeout: int = 120
    debug: Optional[bool] = False
    provider: str = "openrouter"
    model: str = "gpt-4.1"

class SparkOutput(BaseModel):
    stdout: str
    stderr: str
    returncode: int
    duration_in_s: float

class SparkBench(AgentBenchmark[SparkInput, SparkOutput]):
    def __init__(self, params: SparkInput):
        super().__init__(params)

    @override
    async def run_agent(self, task: BenchmarkTask) -> SparkOutput:
        prompt = task.question
        env = os.environ.copy()
        if self.params.env:
            env.update(self.params.env)

        env["SPARK_PROVIDER"] = self.params.provider
        env["SPARK_MODEL"] = self.params.model
        env["SPARK_HEADLESS"] = "true"

        command = ["pnpm", "spark", "run", "--logger", "json", prompt]

        cwd = self.params.spark_dir
        debug = getattr(self.params, "debug", False)
        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                if debug:
                    # Stream output to console and capture
                    async def read_stream(stream, is_stdout=True):
                        chunks = []
                        while True:
                            line = await stream.readline()
                            if not line:
                                break
                            chunks.append(line)
                            print((line.decode(errors="replace").rstrip()), file=(sys.stdout if is_stdout else sys.stderr))
                        return b"".join(chunks)
                    import sys
                    stdout_task = asyncio.create_task(read_stream(proc.stdout, True))
                    stderr_task = asyncio.create_task(read_stream(proc.stderr, False))
                    try:
                        stdout, stderr = await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=self.params.timeout)
                        returncode = proc.returncode
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        stdout, stderr = b"", b"Process timed out."
                        returncode = -1
                else:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.params.timeout)
                    returncode = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"Process timed out."
                returncode = -1
        except Exception as e:
            stdout, stderr = b"", str(e).encode()
            returncode = -2
        duration = time.time() - start_time
        return SparkOutput(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=returncode,
            duration_in_s=duration,
        )

    @override
    async def process_output(self, task: BenchmarkTask, out: SparkOutput) -> TaskResult:
        # Parse JSON events from stdout
        llm_calls: List[LLMCall] = []
        final_answer = ""
        total_duration = out.duration_in_s
        
        # Parse each line as a JSON event
        for line in out.stdout.strip().splitlines():
            try:
                # Skip non-JSON lines (like console output)
                if not line.strip().startswith('{'):
                    continue
                    
                event = json.loads(line)
                event_type = event.get("event")
                data = event.get("data", {})
                
                # Extract LLM call statistics from ai:generation events
                if event_type == "ai:generation" and "usage" in data:
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("promptTokens", 0)
                    completion_tokens = usage.get("completionTokens", 0)
                    total_tokens = usage.get("totalTokens", prompt_tokens + completion_tokens)
                    
                    # Handle both prompt-based and message-based LLM calls
                    messages_in = []
                    
                    # Case 1: Prompt-based LLM call
                    if "prompt" in data and data["prompt"]:
                        prompt = data.get("prompt", "")
                        messages_in = [{"role": "system", "content": prompt}]
                    
                    # Case 2: Message-based LLM call
                    elif "messages" in data and data["messages"]:
                        messages = data.get("messages", [])
                        messages_in = messages
                    
                    # Extract the response object
                    response_obj = data.get("object", {})
                    
                    # Create message_out with proper format
                    if isinstance(response_obj, dict):
                        # If it's a structured response, use it directly
                        content = json.dumps(response_obj)
                    else:
                        # Otherwise convert to string
                        content = str(response_obj)
                        
                    message_out = {"role": "assistant", "content": content}
                    
                    # Create pretty_out (formatted response)
                    pretty_out = json.dumps(response_obj, indent=2) if isinstance(response_obj, dict) else str(response_obj)
                    
                    # Create and add LLMCall
                    llm_call = LLMCall(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        messages_in=messages_in,
                        message_out=message_out,
                        pretty_out=pretty_out
                    )
                    llm_calls.append(llm_call)
                
                # Extract final answer from task:complete event
                elif event_type == "task:complete":
                    final_answer = data.get("finalAnswer", "")
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
            except Exception as e:
                # Log any other errors but continue processing
                print(f"Error processing event: {e}")
        
        # If we didn't find a final answer in task:complete, use the last line as fallback
        if not final_answer:
            stdout_lines = out.stdout.strip().splitlines()
            final_answer = stdout_lines[-1] if stdout_lines else ""
        
        # Create the step with all LLM calls already included
        steps = [Step(url=task.url or "", duration_in_s=total_duration, llm_calls=llm_calls)]
        
        # Determine success based on return code and presence of an answer
        success = out.returncode == 0 and bool(final_answer)
        
        from notte.utils.webp_replay import ScreenshotReplay
        return TaskResult(
            success=success,
            duration_in_s=total_duration,
            agent_answer=final_answer,
            task=task,
            steps=steps,
            screenshots=ScreenshotReplay.from_base64([]),
        )
