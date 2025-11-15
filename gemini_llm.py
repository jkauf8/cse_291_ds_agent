"""Custom Gemini LLM wrapper for LangChain compatibility."""
import os
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from typing import Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM(LLM):
    """Custom LLM wrapper for Google Gemini models."""
    
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configure the API
        api_key = kwargs.get('api_key') or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self._genai_model = genai.GenerativeModel(self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API."""
        try:
            response = self._genai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

