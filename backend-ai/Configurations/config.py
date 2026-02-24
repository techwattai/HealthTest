from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    DEPLOYMENT: str
    DEPLOYMENT: str
    MAIL_USERNAME: str | None = None # Mail username
    MAIL_PASSWORD: str | None = None
    MAIL_FROM: str | None = None
    MAIL_PORT: int | None = 465
    MAIL_SERVER: str | None = None


    class Config:
        env_file = ".env"
        encoding = "utf-8"
        case_sensitive = True

settings = Settings()
AI_VERSION = "v2.0.0"  # Clinical Documentation & Tracking module
# print(settings.OPENAI_API_KEY  )  # Test to ensure settings are loaded correctly
# print(settings.OPENAI_MODEL )  # Test to ensure settings are loaded correctly

class LLMSetup:
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL
    
    def LLM(self):
        llm = ChatOpenAI(model=self.model_name,openai_api_key=self.api_key)
        return llm

llm_model = LLMSetup()

# test = llm_model.LLM().invoke("Hello, world!")  # Test invocation to ensure setup is correct
# print(test.content)  # Print the response content to verify functionality
    