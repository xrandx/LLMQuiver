from enum import Enum, unique


@unique
class SupportAPI(Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"
    OpenAILike = "openai_like"
