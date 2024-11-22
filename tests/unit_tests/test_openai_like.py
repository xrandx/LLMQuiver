from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm_quiver import LLMQuiver, TomlLLMQuiver
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

# .env file
env_path = Path("tests/unit_tests/configs/.test_openai_like.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)


# setup LLMQUIVER_CONFIG, API_BASE, API_KEY
def test_llm_quiver():
    llm = LLMQuiver()
    prompt_values = ["你是谁啊"]
    responses = llm.generate(prompt_values)
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")


def test_toml_llm_quiver():
    llm = TomlLLMQuiver(
        config_path="tests/unit_tests/configs/vllm.toml",
        toml_prompt_name="hello_world_template",
        toml_template_file="tests/unit_tests/prompts/hello_world.toml"
    )
    prompt_values = [dict(name="GPT")]
    responses = llm.generate(prompt_values=prompt_values)
    logger.info(f"responses: {responses}")
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")

    llm = TomlLLMQuiver(
        config_path="tests/unit_tests/configs/vllm.toml",
        toml_template_file="tests/unit_tests/prompts/messages_test.toml",
        toml_prompt_name="joke_template"
    )
    prompt_values = [dict(word="马斯克")]
    responses = llm.chat(prompt_values=prompt_values)
    logger.info(f"responses: {responses}")
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")


if __name__ == "__main__":
    test_llm_quiver()
    test_toml_llm_quiver()
