import time
from tqdm import tqdm
import warnings
from hugchat import hugchat
from hugchat.login import Login

from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from hugchat import hugchat
from hugchat.login import Login
from Prompts import email, passwd


###########Llama3###########
sign = Login(email, passwd)
cookies = sign.login()

cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)

tqdm.pandas()
warnings.filterwarnings("ignore") 
# CustomLLM
def prediction(prompt: str, temperature: float) -> str:
    try:
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        chatbot.delete_all_conversations()
        chatbot.switch_llm(2)
        chatbot.new_conversation(modelIndex=2, switch_to = True, assistant=None, system_prompt="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf")
        time.sleep(5)
    except:
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        chatbot.switch_llm(2)
        chatbot.new_conversation(modelIndex=2, switch_to = True, assistant=None, system_prompt="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf")
        time.sleep(5)
    response = chatbot.query(text=prompt, temperature=temperature, web_search=False)
    return response['text']

class CustomLLM(LLM):
    temperature: float
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        temperature: Optional[float]=1,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prediction(prompt, temperature)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"temperature": self.temperature}