import time
from typing import ClassVar, Optional

from typing_extensions import override

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai.chat_models import ChatOpenAI
except ImportError:
    raise ImportError("WebVoyager evaluator requires installing langchain_openai")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


from eval.evaluators.evaluator import EvalEnum, EvaluationResponse, Evaluator


class WebvoyagerEvaluator(Evaluator):
    SYSTEM_PROMPT: ClassVar[
        str
    ] = """As an evaluator, you will be presented with three primary components to assist you in your role:

    1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

    2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction, and may not represent everything the agent sees.

    3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

    -- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
    -- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions. If you cannot find any information in the screenshot that matches the instruction, you can believe the information in the response.
    -- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
    -- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
    -- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
    -- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.
    -- If you are not sure whether you should believe the content in the response, you should choose unknown.

    You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS', 'NOT SUCCESS', or 'UNKNOWN'."""

    USER_PROMPT: ClassVar[str] = """TASK: <task>
    Result Response: <answer>
    <num> screenshot at the end: """

    past_screenshots: int = 4
    tries: int = 3
    model: str = "gpt-4o"
    provider: str = "openai"  # "openai" or "google"
    google_credentials_path: Optional[str] = None
    google_project_id: Optional[str] = None

    def _create_llm_client(self):
        """Create LLM client based on configured provider."""
        if self.provider == "google":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("Google AI support requires installing langchain-google-genai")
            
            # Configure Google AI client
            kwargs = {"model": self.model}
            
            # Set credentials if explicitly provided and file exists (for local development)
            if self.google_credentials_path:
                import os
                if os.path.exists(self.google_credentials_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_credentials_path
                else:
                    # File doesn't exist, let it fall back to ADC
                    print(f"Warning: Credentials file {self.google_credentials_path} not found, using Application Default Credentials")
            # Otherwise uses Application Default Credentials (ADC) - perfect for Cloud Tasks!
            
            if self.google_project_id:
                kwargs["project_id"] = self.google_project_id
                
            return ChatGoogleGenerativeAI(**kwargs)
        
        elif self.provider == "openai":
            return ChatOpenAI(model=self.model)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported: 'openai', 'google'")

    @override
    async def eval(
        self,
        answer: str,
        task: str,
        screenshots: list[str],
    ) -> EvaluationResponse:
        # Create LLM client based on provider
        llm = self._create_llm_client()

        screenshots = screenshots[-self.past_screenshots :]
        screenshot_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}"},
            }
            for screenshot in screenshots
        ]

        # Prepare GPT-4V messages
        user_prompt_tmp = WebvoyagerEvaluator.USER_PROMPT.replace("<task>", task)
        user_prompt_tmp = user_prompt_tmp.replace("<answer>", answer)
        user_prompt_tmp = user_prompt_tmp.replace("<num>", str(len(screenshots)))

        messages = [
            SystemMessage(content=WebvoyagerEvaluator.SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt_tmp},
                    *screenshot_content,
                    {"type": "text", "text": "Your verdict:\n"},
                ]
            ),
        ]

        gpt_4v_res = "Failed to get response"
        tries = self.tries
        while tries >= 0:
            try:
                tries -= 1
                # print("Calling gpt4v API to get the auto evaluation......")
                response = await llm.ainvoke(messages)
                gpt_4v_res = str(response.content)  # type: ignore
                break
            except Exception as e:
                print(e)
                error_name = type(e).__name__
                # Handle OpenAI errors
                if error_name == "RateLimitError":
                    time.sleep(10)
                elif error_name == "APIError":
                    time.sleep(15)
                elif error_name == "InvalidRequestError":
                    exit(0)
                # Handle Google AI errors
                elif error_name == "ResourceExhausted":  # Google AI rate limit
                    time.sleep(10)
                elif error_name == "InvalidArgument":  # Google AI invalid request
                    exit(0)
                elif error_name == "PermissionDenied":  # Google AI auth error
                    print(f"Google AI authentication error: {e}")
                    exit(0)
                elif error_name == "NotFound":  # Google AI model not found
                    print(f"Google AI model not found: {e}")
                    exit(0)
                else:
                    time.sleep(10)

        if "NOT SUCCESS" in gpt_4v_res:
            auto_eval_res = EvalEnum.FAILURE
        elif "SUCCESS" in gpt_4v_res:
            auto_eval_res = EvalEnum.SUCCESS
        elif "UNKNOWN" in gpt_4v_res:
            auto_eval_res = EvalEnum.UNKNOWN
        else:
            auto_eval_res = EvalEnum.FAILURE

        return EvaluationResponse(eval=auto_eval_res, reason=gpt_4v_res)
