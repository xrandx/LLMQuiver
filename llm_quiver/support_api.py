from enum import Enum, unique


@unique
class SupportAPI(Enum):
    AzureOpenAI = "azure_openai"
    OpenAI = "openai"
    OpenAILike = "openai_like"


api_type = "azure_openai"
try:
    print(SupportAPI(api_type))
except KeyError:
    raise ValueError(f"{api_type} is not a valid name for {SupportAPI.__name__}")
