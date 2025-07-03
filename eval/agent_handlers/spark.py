import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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


def prepare_environment(params: SparkInput) -> Dict[str, str]:
    """Prepare environment variables for Spark execution."""
    env = os.environ.copy()
    if params.env:
        env.update(params.env)

    # Set Spark-specific environment variables
    env["SPARK_PROVIDER"] = params.provider
    env["SPARK_MODEL"] = params.model
    env["SPARK_HEADLESS"] = "true"
    env["SPARK_LOGGER"] = "json"

    # Copy proxy settings if available
    proxy_vars = ["SPARK_PROXY", "SPARK_PROXY_USERNAME", "SPARK_PROXY_PASSWORD"]
    for var in proxy_vars:
        if var in os.environ:
            env[var] = os.environ[var]

    # Copy API keys
    api_keys = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_CLOUD_PROJECT"]
    for key in api_keys:
        if key in os.environ:
            env[key] = os.environ[key]

    return env


def build_command(url: str, prompt: str) -> List[str]:
    """Build the Spark command to execute."""
    return ["npm", "run", "spark", "--", "run", "--url", url, prompt]


async def read_stream(stream, is_stdout: bool, start_time: float, debug: bool) -> bytes:
    """Read from a stream with proper buffering and optional debug output."""
    chunks = []
    line_count = 0
    buffer = b""

    try:
        while True:
            # Read larger chunks to handle big AI responses
            chunk = await stream.read(65536)  # 64KB chunks
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
                                print(f"[{stream_name}:final@{time.time()-start_time:.1f}s] {buffer_text[-200:]}", 
                                      file=(sys.stdout if is_stdout else sys.stderr))
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
                    print_debug_line(line, line_count, current_time - start_time, is_stdout)
                else:
                    print_progress_event(line, is_stdout)

    except Exception as e:
        # If streaming fails, add error info but don't fail completely
        error_msg = f"Stream reading error ({'stdout' if is_stdout else 'stderr'}): {e}\n"
        chunks.append(error_msg.encode())
        if debug:
            print(f"SPARK DEBUG: Stream reading exception: {e}")

    return b"".join(chunks)


def print_debug_line(line: bytes, line_count: int, elapsed: float, is_stdout: bool):
    """Print a debug line with proper formatting."""
    stream_name = "stdout" if is_stdout else "stderr"
    try:
        line_text = line.decode(errors="replace").rstrip()
        # Truncate very long lines to prevent buffering issues
        if len(line_text) > MAX_DEBUG_LINE_LENGTH:
            line_text = line_text[:MAX_DEBUG_LINE_LENGTH] + "..."
        print(f"[{stream_name}:{line_count}@{elapsed:.1f}s] {line_text}", 
              file=(sys.stdout if is_stdout else sys.stderr))
    except Exception:
        # Don't let decode errors break stream processing
        pass


def print_progress_event(line: bytes, is_stdout: bool):
    """Print progress for key events in non-debug mode."""
    if not is_stdout:
        return

    try:
        line_text = line.decode(errors="replace").rstrip()
        if not line_text.startswith('{'):
            return

        event = json.loads(line_text)
        event_type = event.get("event", "")
        data = event.get("data", {})

        # Show progress for key events
        progress_messages = {
            "task:setup": lambda: f"🚀 Starting task: {str(data.get('task', ''))[:50]}...",
            "task:started": lambda: f"📋 Task started at: {data.get('url', '')}",
            "browser:navigated": lambda: f"🌐 Navigated to: {data.get('url', '')}",
            "agent:step": lambda: f"🤖 Step: {data.get('currentStep', '')}",
            "agent:observed": lambda: f"👀 Observed: {str(data.get('observation', ''))[:100]}...",
            "browser:action_started": lambda: f"⚡ Action: {data.get('action', '')} {data.get('ref', '')}",
            "browser:action_completed": lambda: f"{'✅' if data.get('success', False) else '❌'} Action completed",
            "task:completed": lambda: "🏁 Task completed"
        }

        if event_type in progress_messages:
            print(progress_messages[event_type](), flush=True)

    except (json.JSONDecodeError, Exception):
        pass


async def run_spark_process(command: List[str], cwd: str, env: Dict[str, str], 
                           timeout: int, debug: bool) -> Tuple[bytes, bytes, int, float]:
    """Run the Spark subprocess and capture output."""
    start_time = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_task = asyncio.create_task(read_stream(proc.stdout, True, start_time, debug))
        stderr_task = asyncio.create_task(read_stream(proc.stderr, False, start_time, debug))

        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task), 
                timeout=timeout
            )
            await proc.wait()  # Ensure process has terminated
            returncode = proc.returncode if proc.returncode is not None else -3

        except asyncio.TimeoutError:
            if debug:
                print(f"SPARK DEBUG: Process timed out after {timeout} seconds")
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

    except Exception as e:
        if debug:
            print(f"SPARK DEBUG: Exception during subprocess creation: {e}")
        stdout, stderr = b"", str(e).encode()
        returncode = -2

    duration = time.time() - start_time

    if debug:
        if returncode != 0:
            print(f"SPARK DEBUG: Process failed with exit code {returncode}")
            if stderr:
                print(f"SPARK DEBUG: Stderr: {stderr.decode(errors='replace')[:200]}")
        else:
            print(f"SPARK DEBUG: Process completed successfully")

    return stdout, stderr, returncode, duration


def parse_json_event(line: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON event from a line of output."""
    if not line.strip().startswith('{'):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def extract_llm_call(event: Dict[str, Any]) -> Optional[LLMCall]:
    """Extract an LLM call from an ai:generation event."""
    if event.get("event") != "ai:generation" or "usage" not in event.get("data", {}):
        return None

    data = event.get("data", {})
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
        messages_in = data.get("messages", [])

    # Extract the response object
    response_obj = data.get("object", {})

    # Create message_out with proper format
    if isinstance(response_obj, dict):
        content = json.dumps(response_obj)
    else:
        content = str(response_obj)

    message_out = {"role": "assistant", "content": content}

    # Create pretty_out (formatted response)
    pretty_out = json.dumps(response_obj, indent=2) if isinstance(response_obj, dict) else str(response_obj)

    return LLMCall(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        messages_in=messages_in,
        message_out=message_out,
        pretty_out=pretty_out
    )


def initialize_step(url: str, timestamp: int, operation: str = "") -> Dict[str, Any]:
    """Initialize a new step data structure."""
    return {
        "url": url,
        "llm_calls": [],
        "start_time": timestamp,
        "end_time": 0,
        "thinking_operation": operation
    }


def extract_error_details(stdout: str, stderr: str, returncode: int) -> str:
    """Extract error details from process output."""
    error_details = []

    # Look for specific error messages in stderr first
    if stderr:
        stderr_lines = stderr.strip().splitlines()
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
    if stdout:
        stdout_lines = stdout.strip().splitlines()
        for line in stdout_lines:
            if "Error:" in line or "❌" in line:
                error_details.append(line.strip())

    # If we found specific error details, use them
    if error_details:
        return f"Task failed with error: {'; '.join(error_details[:3])}"  # Limit to first 3 error lines
    else:
        # Fallback to generic error message with return code
        return f"Task failed with exit code {returncode}. Check stderr for details."


def extract_final_answer(stdout: str) -> str:
    """Extract the final answer from stdout."""
    stdout_lines = stdout.strip().splitlines()
    
    for line in reversed(stdout_lines):
        event = parse_json_event(line)
        if not event:
            continue
            
        if event.get("event") == "browser:action_completed" and event.get("data", {}).get("success"):
            return "Task completed successfully based on browser actions"
        elif event.get("event") == "agent:extracted" and event.get("data", {}).get("extractedData"):
            return str(event.get("data", {}).get("extractedData", ""))

    # Final fallback to last line if nothing else found
    return stdout_lines[-1] if stdout_lines else ""


def convert_steps_to_objects(steps_data: List[Dict[str, Any]], total_duration: float) -> List[Step]:
    """Convert step data dictionaries to Step objects."""
    steps = []
    for step_data in steps_data:
        duration = (step_data["end_time"] - step_data["start_time"]) / 1000  # Convert ms to seconds
        duration = max(0.001, duration)  # Ensure positive duration
        steps.append(Step(
            url=step_data["url"],
            duration_in_s=duration,
            llm_calls=step_data["llm_calls"]
        ))
    return steps


class SparkBench(AgentBenchmark[SparkInput, SparkOutput]):
    def __init__(self, params: SparkInput):
        super().__init__(params)

    @override
    async def run_agent(self, task: BenchmarkTask) -> SparkOutput:
        url = task.url
        prompt = task.question
        
        env = prepare_environment(self.params)
        command = build_command(url, prompt)
        
        stdout, stderr, returncode, duration = await run_spark_process(
            command=command,
            cwd=self.params.spark_dir,
            env=env,
            timeout=self.params.timeout,
            debug=self.params.debug
        )
        
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
        debug_info = {
            "total_lines": 0,
            "json_lines": 0,
            "ai_generation_events": 0,
            "agent_thinking_events": 0,
            "stdout_length": len(out.stdout),
            "stderr_length": len(out.stderr),
            "stdout_preview": out.stdout[:200],
            "stderr_preview": out.stderr[:200],
            "return_code": out.returncode,
            "duration": out.duration_in_s,
            "command": "npm run spark"
        }
        
        # Parse each line as a JSON event - check both stdout and stderr
        all_output = out.stdout + "\n" + out.stderr
        lines = all_output.strip().splitlines()
        
        for line in lines:
            debug_info["total_lines"] += 1
            event = parse_json_event(line)
            if not event:
                debug_info["non_json_lines"] = debug_info.get("non_json_lines", 0) + 1
                continue
            
            debug_info["json_lines"] += 1
            event_type = event.get("event")
            
            # Count event types for debugging
            if event_type:
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            data = event.get("data", {})
            timestamp = event.get("timestamp", 0)
            
            # Initialize the first step when we see task:setup
            if event_type == "task:setup" and current_step is None:
                current_step = initialize_step(
                    url=data.get("url", task.url or ""),
                    timestamp=timestamp,
                    operation="Initial setup"
                )
            
            elif event_type == "task:started":
                if current_step is not None:
                    current_step["url"] = data.get("url", task.url or "")
            
            # Start a new step when we see an agent:processing event with status "start"
            elif event_type == "agent:processing":
                debug_info["agent_thinking_events"] += 1
                
                # Initialize current_step if it doesn't exist yet
                if current_step is None:
                    current_step = initialize_step(
                        url=task.url or "",
                        timestamp=timestamp,
                        operation=data.get("operation", "")
                    )
                
                if data.get("status") == "start":
                    # Save the previous step if it has content
                    if current_step["llm_calls"] or current_step.get("has_navigation", False):
                        current_step["end_time"] = timestamp
                        steps_data.append(current_step)
                    
                    # Start a new step
                    current_step = initialize_step(
                        url=current_step["url"],  # Inherit the URL from the previous step
                        timestamp=timestamp,
                        operation=data.get("operation", "")
                    )
                elif data.get("status") == "end" and current_step:
                    current_step["end_time"] = timestamp
            
            # Update the URL when we see a browser:navigated event
            elif event_type == "browser:navigated" and "url" in data:
                # Initialize current_step if it doesn't exist yet
                if current_step is None:
                    current_step = initialize_step(
                        url=data.get("url", task.url or ""),
                        timestamp=timestamp,
                        operation="Navigation"
                    )
                else:
                    current_step["url"] = data.get("url", current_step["url"])
                    current_step["has_navigation"] = True
            
            # Extract LLM call statistics from ai:generation events
            elif event_type == "ai:generation" and "usage" in data:
                debug_info["ai_generation_events"] += 1
                
                # Initialize current_step if it doesn't exist yet
                if current_step is None:
                    current_step = initialize_step(
                        url=task.url or "",
                        timestamp=timestamp,
                        operation="AI Generation"
                    )
                
                llm_call = extract_llm_call(event)
                if llm_call:
                    # Add to current step and to all LLM calls list
                    current_step["llm_calls"].append(llm_call)
                    all_llm_calls.append(llm_call)
            
            # Extract final answer from task:complete event
            elif event_type == "task:completed":
                final_answer = data.get("finalAnswer", "")
                # Mark the end of the current step if it exists
                if current_step and current_step["end_time"] == 0:
                    current_step["end_time"] = timestamp
        
        # Add the last step if it exists and hasn't been added yet
        if current_step and (current_step["llm_calls"] or current_step.get("has_navigation", False)):
            if current_step["end_time"] == 0:
                current_step["end_time"] = current_step["start_time"] + 1000  # Default 1 second duration if no end time
            steps_data.append(current_step)
        
        # If we didn't find a final answer in task:complete, try to extract from final actions
        if not final_answer:
            final_answer = extract_final_answer(out.stdout)
        
        # Convert step data to Step objects
        steps = convert_steps_to_objects(steps_data, total_duration)
        
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
            if not final_answer:
                final_answer = extract_error_details(out.stdout, out.stderr, out.returncode)
        
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