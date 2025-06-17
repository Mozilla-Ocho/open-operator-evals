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
                        await proc.wait()  # Ensure process has terminated
                        returncode = proc.returncode if proc.returncode is not None else -3
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        stdout, stderr = b"", b"Process timed out."
                        returncode = -1
                else:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.params.timeout)
                    returncode = proc.returncode if proc.returncode is not None else -3
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
        all_llm_calls = []  # Collect all LLM calls in case step tracking fails
        steps_data = []
        current_step = None
        final_answer = ""
        total_duration = out.duration_in_s
        
        # Debug counters
        event_counts = {}
        debug_info = {"total_lines": 0, "json_lines": 0, "ai_generation_events": 0, "agent_thinking_events": 0}
        
        # Parse each line as a JSON event
        for line in out.stdout.strip().splitlines():
            debug_info["total_lines"] += 1
            try:
                # Skip non-JSON lines (like console output)
                if not line.strip().startswith('{'):
                    continue
                
                debug_info["json_lines"] += 1
                event = json.loads(line)
                event_type = event.get("event")
                
                # Count event types for debugging
                if event_type:
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                data = event.get("data", {})
                timestamp = event.get("timestamp", 0)
                
                # Initialize the first step when we see task:setup
                if event_type == "task:setup" and current_step is None:
                    current_step = {
                        "url": data.get("url", task.url or ""),
                        "llm_calls": [],
                        "start_time": timestamp,
                        "end_time": 0,
                        "thinking_operation": "Initial setup"
                    }
                
                # Start a new step when we see an agent:thinking event with status "start"
                elif event_type == "agent:thinking":
                    debug_info["agent_thinking_events"] += 1
                    
                    # Initialize current_step if it doesn't exist yet
                    if current_step is None:
                        current_step = {
                            "url": task.url or "",
                            "llm_calls": [],
                            "start_time": timestamp,
                            "end_time": 0,
                            "thinking_operation": data.get("operation", "")
                        }
                    
                    if data.get("status") == "start":
                        # Save the previous step if it has content
                        if current_step["llm_calls"] or current_step.get("has_navigation", False):
                            current_step["end_time"] = timestamp
                            steps_data.append(current_step)
                        
                        # Start a new step
                        current_step = {
                            "url": current_step["url"],  # Inherit the URL from the previous step
                            "llm_calls": [],
                            "start_time": timestamp,
                            "end_time": 0,
                            "thinking_operation": data.get("operation", "")
                        }
                    elif data.get("status") == "end" and current_step:
                        current_step["end_time"] = timestamp
                
                # Update the URL when we see a page:navigation event
                elif event_type == "page:navigation" and "url" in data:
                    # Initialize current_step if it doesn't exist yet
                    if current_step is None:
                        current_step = {
                            "url": data.get("url", task.url or ""),
                            "llm_calls": [],
                            "start_time": timestamp,
                            "end_time": 0,
                            "thinking_operation": "Navigation"
                        }
                    else:
                        current_step["url"] = data.get("url", current_step["url"])
                        current_step["has_navigation"] = True
                
                # Extract LLM call statistics from ai:generation events
                elif event_type == "ai:generation" and "usage" in data:
                    debug_info["ai_generation_events"] += 1
                    
                    # Initialize current_step if it doesn't exist yet
                    if current_step is None:
                        current_step = {
                            "url": task.url or "",
                            "llm_calls": [],
                            "start_time": timestamp,
                            "end_time": 0,
                            "thinking_operation": "AI Generation"
                        }
                    
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("promptTokens", 0)
                    completion_tokens = usage.get("completionTokens", 0)
                    
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
                    
                    # Add to current step and to all LLM calls list
                    current_step["llm_calls"].append(llm_call)
                    all_llm_calls.append(llm_call)
                
                # Extract final answer from task:complete event
                elif event_type == "task:complete":
                    final_answer = data.get("finalAnswer", "")
                    # Mark the end of the current step if it exists
                    if current_step and current_step["end_time"] == 0:
                        current_step["end_time"] = timestamp
                    
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
            except Exception as e:
                # Log any other errors but continue processing
                print(f"Error processing event: {e}")
        
        # Add the last step if it exists and hasn't been added yet
        if current_step and (current_step["llm_calls"] or current_step.get("has_navigation", False)):
            if current_step["end_time"] == 0:
                current_step["end_time"] = current_step["start_time"] + 1000  # Default 1 second duration if no end time
            steps_data.append(current_step)
        
        # If we didn't find a final answer in task:complete, use the last line as fallback
        if not final_answer:
            stdout_lines = out.stdout.strip().splitlines()
            final_answer = stdout_lines[-1] if stdout_lines else ""
        
        # Convert step data to Step objects
        steps = []
        for step_data in steps_data:
            duration = (step_data["end_time"] - step_data["start_time"]) / 1000  # Convert ms to seconds
            duration = max(0.001, duration)  # Ensure positive duration
            steps.append(Step(
                url=step_data["url"],
                duration_in_s=duration,
                llm_calls=step_data["llm_calls"]
            ))
        
        # If no steps were created but we have LLM calls, create a single step with all LLM calls
        if not steps and all_llm_calls:
            steps = [Step(url=task.url or "", duration_in_s=total_duration, llm_calls=all_llm_calls)]
        # If no steps or LLM calls, create an empty step
        elif not steps:
            steps = [Step(url=task.url or "", duration_in_s=total_duration, llm_calls=[])] 
        
        # Determine success based on return code and presence of an answer
        success = out.returncode == 0 and bool(final_answer)
        
        # Add debug info to logs
        logs = {
            "debug_info": json.dumps(debug_info),
            "event_counts": json.dumps(event_counts),
            "llm_call_count": str(len(all_llm_calls)),
            "step_count": str(len(steps)),
            "raw_steps": json.dumps([{"url": s["url"], "operation": s["thinking_operation"], "llm_calls": len(s["llm_calls"])} for s in steps_data])
        }
        
        from notte.utils.webp_replay import ScreenshotReplay
        return TaskResult(
            success=success,
            duration_in_s=total_duration,
            agent_answer=final_answer,
            task=task,
            steps=steps,
            logs=logs,
            screenshots=ScreenshotReplay.from_base64([]),
        )
