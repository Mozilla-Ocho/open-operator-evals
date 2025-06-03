import asyncio
import os
import time
from typing import Any, Dict, Optional

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
        env["LLM_PROVIDER"] = self.params.provider
        env["LLM_MODEL"] = self.params.model
        command = ["pnpm", "run", "spark", prompt]
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
                    stdout_chunks = []
                    stderr_chunks = []
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
        # For now, treat the last line of stdout as the answer, and collect all output as a single step.
        steps = []
        llm_calls = []
        stdout_lines = out.stdout.strip().splitlines()
        answer = stdout_lines[-1] if stdout_lines else ""
        # Optionally, parse step-by-step output if Spark provides it in the future
        steps.append(Step(url=task.url or "", duration_in_s=out.duration_in_s, llm_calls=llm_calls))
        success = out.returncode == 0 and bool(answer)
        from notte.utils.webp_replay import ScreenshotReplay
        return TaskResult(
            success=success,
            duration_in_s=out.duration_in_s,
            agent_answer=answer,
            task=task,
            steps=steps,
            screenshots=ScreenshotReplay.from_base64([]),
        )
