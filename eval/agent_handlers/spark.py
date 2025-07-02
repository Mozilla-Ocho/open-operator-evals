import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import override

from eval.data.load_data import BenchmarkTask
from eval.task_types import AgentBenchmark, Step, TaskResult, LLMCall

MAX_DEBUG_LINE_LENGTH = 500  # Maximum length of debug lines to prevent excessive buffering

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
        url = task.url
        prompt = task.question
        env = os.environ.copy()
        if self.params.env:
            env.update(self.params.env)

        env["SPARK_PROVIDER"] = self.params.provider
        env["SPARK_MODEL"] = self.params.model
        env["SPARK_HEADLESS"] = "true"
        env["SPARK_LOGGER"] = "json"
        
        # Set API keys for Spark
        if "OPENROUTER_API_KEY" in os.environ:
            env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        if "OPENAI_API_KEY" in os.environ:
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        
        if "SPARK_PROXY" in os.environ:
            env["SPARK_PROXY"] = os.environ["SPARK_PROXY"]
        if "SPARK_PROXY_USERNAME" in os.environ:
            env["SPARK_PROXY_USERNAME"] = os.environ["SPARK_PROXY_USERNAME"] 
        if "SPARK_PROXY_PASSWORD" in os.environ:
            env["SPARK_PROXY_PASSWORD"] = os.environ["SPARK_PROXY_PASSWORD"]

        debug = getattr(self.params, "debug", False)

        command = ["npm", "run", "spark", "--", "run", "--url", url, prompt]

        cwd = self.params.spark_dir
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
                # Enhanced streaming with large buffer support
                async def read_stream(stream, is_stdout=True):
                    chunks = []
                    line_count = 0 
                    buffer = b""
                    
                    try:
                        while True:
                            # Read larger chunks to handle big AI responses
                            chunk = await stream.read(65536)  # 64KB chunks instead of readline
                            if not chunk:
                                # Process any remaining buffer content
                                if buffer:
                                    chunks.append(buffer)
                                    if debug:
                                        stream_name = "stdout" if is_stdout else "stderr"
                                        try:
                                            buffer_text = buffer.decode(errors="replace").rstrip()
                                            if buffer_text:
                                                line_count += buffer_text.count('\n') + 1
                                                print(f"[{stream_name}:final@{time.time()-start_time:.1f}s] {buffer_text[-200:]}", file=(sys.stdout if is_stdout else sys.stderr))
                                        except Exception:
                                            pass
                                break
                                
                            buffer += chunk
                            current_time = time.time()
                            
                            # Process complete lines from buffer
                            while b'\n' in buffer:
                                line, buffer = buffer.split(b'\n', 1)
                                line += b'\n'  # Add back the newline
                                chunks.append(line)
                                line_count += 1
                                
                                if debug:
                                    stream_name = "stdout" if is_stdout else "stderr"
                                    try:
                                        line_text = line.decode(errors="replace").rstrip()
                                        # Truncate very long lines to prevent buffering issues
                                        if len(line_text) > MAX_DEBUG_LINE_LENGTH:
                                            line_text = line_text[:MAX_DEBUG_LINE_LENGTH] + "..."
                                        print(f"[{stream_name}:{line_count}@{current_time-start_time:.1f}s] {line_text}", file=(sys.stdout if is_stdout else sys.stderr))
                                    except Exception:
                                        # Don't let decode errors break stream processing
                                        pass
                                else:
                                    # Show progress for key events even in non-debug mode
                                    try:
                                        line_text = line.decode(errors="replace").rstrip()
                                        if line_text.startswith('{') and is_stdout:
                                            try:
                                                event = json.loads(line_text)
                                                event_type = event.get("event", "")
                                                data = event.get("data", {})
                                                
                                                # Show progress for key events
                                                if event_type in ["task:setup", "task:started", "browser:navigated", "agent:step", "agent:observed", "browser:action_started", "browser:action_completed", "task:completed"]:
                                                    if event_type == "task:setup":
                                                        task_name = str(data.get('task', ''))[:50]
                                                        print(f"🚀 Starting task: {task_name}...", flush=True)
                                                    elif event_type == "task:started":
                                                        url = str(data.get('url', ''))
                                                        print(f"📋 Task started at: {url}", flush=True)
                                                    elif event_type == "browser:navigated":
                                                        url = str(data.get('url', ''))
                                                        print(f"🌐 Navigated to: {url}", flush=True)
                                                    elif event_type == "agent:step":
                                                        step = str(data.get('currentStep', ''))
                                                        print(f"🤖 Step: {step}", flush=True)
                                                    elif event_type == "agent:observed":
                                                        obs = str(data.get('observation', ''))[:100]
                                                        print(f"👀 Observed: {obs}...", flush=True)
                                                    elif event_type == "browser:action_started":
                                                        action = str(data.get('action', ''))
                                                        ref = str(data.get('ref', ''))
                                                        print(f"⚡ Action: {action} {ref}", flush=True)
                                                    elif event_type == "browser:action_completed":
                                                        success = data.get('success', False)
                                                        status = "✅" if success else "❌"
                                                        print(f"{status} Action completed", flush=True)
                                                    elif event_type == "task:completed":
                                                        print(f"🏁 Task completed", flush=True)
                                            except json.JSONDecodeError:
                                                pass
                                    except Exception:
                                        pass
                            
                    except Exception as e:
                        # If streaming fails, add error info but don't fail completely
                        error_msg = f"Stream reading error ({'stdout' if is_stdout else 'stderr'}): {e}\n"
                        chunks.append(error_msg.encode())
                        if debug:
                            print(f"SPARK DEBUG: Stream reading exception: {e}")
                    
                    return b"".join(chunks)
                
                stdout_task = asyncio.create_task(read_stream(proc.stdout, True))
                stderr_task = asyncio.create_task(read_stream(proc.stderr, False))
                try:
                    stdout, stderr = await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=self.params.timeout)
                    await proc.wait()  # Ensure process has terminated
                    returncode = proc.returncode if proc.returncode is not None else -3
                        
                except asyncio.TimeoutError:
                    if debug:
                        print(f"SPARK DEBUG: Process timed out after {self.params.timeout} seconds")
                    # Cancel the stream reading tasks to prevent hanging
                    stdout_task.cancel()
                    stderr_task.cancel()
                    proc.kill()
                    await proc.wait()
                    # Try to get partial output from completed tasks
                    try:
                        stdout = await stdout_task
                    except (asyncio.CancelledError, Exception):
                        stdout = b""
                    try:
                        stderr = await stderr_task
                    except (asyncio.CancelledError, Exception):
                        stderr = b"Process timed out."
                    returncode = -1
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"Process timed out."
                returncode = -1
        except Exception as e:
            if debug:
                print(f"SPARK DEBUG: Exception during subprocess creation: {e}")
            stdout, stderr = b"", str(e).encode()
            returncode = -2
        duration = time.time() - start_time
        
        if returncode != 0 and debug:
            print(f"SPARK DEBUG: Process failed with exit code {returncode}")
            if stderr:
                print(f"SPARK DEBUG: Stderr: {stderr.decode(errors='replace')[:200]}")
        elif debug:
            print(f"SPARK DEBUG: Process completed successfully")
        
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
        
        # Parse each line as a JSON event - check both stdout and stderr
        all_output = out.stdout + "\n" + out.stderr
        lines = all_output.strip().splitlines()
        
        # Debug: Save actual output for debugging
        debug_info["stdout_length"] = len(out.stdout)
        debug_info["stderr_length"] = len(out.stderr)
        debug_info["stdout_preview"] = out.stdout[:200]
        debug_info["stderr_preview"] = out.stderr[:200]
        debug_info["return_code"] = out.returncode
        debug_info["duration"] = out.duration_in_s
        debug_info["command"] = "npm run spark"  # Fixed: command not in scope
        
        for line in lines:
            debug_info["total_lines"] += 1
            try:
                # Skip non-JSON lines (like console output)
                if not line.strip().startswith('{'):
                    debug_info["non_json_lines"] = debug_info.get("non_json_lines", 0) + 1
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
                
                elif event_type == "task:started":
                    if current_step is not None:
                        current_step["url"] = data.get("url", task.url or "")
                
                # Start a new step when we see an agent:processing event with status "start"
                elif event_type == "agent:processing":
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
                
                # Update the URL when we see a browser:navigated event
                elif event_type == "browser:navigated" and "url" in data:
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
                elif event_type == "task:completed":
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
        
        # If we didn't find a final answer in task:complete, try to extract from final actions
        if not final_answer:
            # Look for successful browser actions that might indicate task completion
            stdout_lines = out.stdout.strip().splitlines()
            for line in reversed(stdout_lines):
                try:
                    if not line.strip().startswith('{'):
                        continue
                    event = json.loads(line)
                    if event.get("event") == "browser:action_completed" and event.get("data", {}).get("success"):
                        final_answer = "Task completed successfully based on browser actions"
                        break
                    elif event.get("event") == "agent:extracted" and event.get("data", {}).get("extractedData"):
                        final_answer = str(event.get("data", {}).get("extractedData", ""))
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Final fallback to last line if nothing else found
            if not final_answer:
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
        
        # When process fails, enhance the agent answer with error details
        if out.returncode != 0:
            error_details = []
            
            # Look for specific error messages in stderr first
            if out.stderr:
                stderr_lines = out.stderr.strip().splitlines()
                for line in stderr_lines:
                    if any(error_type in line for error_type in [
                        "NS_ERROR_PROXY_FORBIDDEN", 
                        "NS_ERROR_", 
                        "Error:", 
                        "TypeError:", 
                        "ReferenceError:",
                        "page.goto:",
                        "Connection refused",
                        "timeout"
                    ]):
                        error_details.append(line.strip())
            
            # Also check stdout for error events
            if out.stdout:
                stdout_lines = out.stdout.strip().splitlines()
                for line in stdout_lines:
                    if "Error:" in line or "❌" in line:
                        error_details.append(line.strip())
            
            # If we found specific error details, use them as the agent answer
            if error_details:
                final_answer = f"Task failed with error: {'; '.join(error_details[:3])}"  # Limit to first 3 error lines
            elif not final_answer:
                # Fallback to generic error message with return code
                final_answer = f"Task failed with exit code {out.returncode}. Check stderr for details."
        
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
