from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm_quiver import LLMQuiver, TomlLLMQuiver
from loguru import logger
from dotenv import load_dotenv

# .env file
env_path = Path("tests/unit_tests/configs/.test.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f"{env_path} not found")


def test_llm_quiver():
    #   test generate
    llm = LLMQuiver()
    prompt_values = ["你是谁啊"]
    responses = llm.generate(prompt_values)
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")

    #   test chat
    prompt_values = [[{"role": "user", "content": "你是谁啊"}]]
    responses = llm.chat(prompt_values)
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")


def test_toml_llm_quiver():
    #   test generate
    llm = TomlLLMQuiver(
        toml_prompt_name="hello_world_template",
        toml_template_file="tests/unit_tests/prompts/hello_world.toml"
    )
    prompt_values = [dict(name="GPT")]
    responses = llm.generate(prompt_values=prompt_values)
    logger.info(f"responses: {responses}")
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")

    #   test chat
    llm = TomlLLMQuiver(
        toml_prompt_name="joke_template",
        toml_template_file="tests/unit_tests/prompts/messages_test.toml"
    )
    prompt_values = [dict(word="川普")]
    responses = llm.chat(prompt_values=prompt_values)
    logger.info(f"responses: {responses}")
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")


def test_no_env():
    import os
    os.environ.pop('LLMQUIVER_CONFIG', None)
    llm = TomlLLMQuiver(
        config_path="tests/unit_tests/configs/gpt4o_with_key.toml",
        toml_prompt_name="hello_world_template",
        toml_template_file="tests/unit_tests/prompts/hello_world.toml"
    )
    prompt_values = [dict(name="GPT")]
    responses = llm.generate(prompt_values=prompt_values)
    logger.info(f"responses: {responses}")
    logger.info(f"responses type: {type(responses)}, {type(responses[0])}")


if __name__ == "__main__":
    test_llm_quiver()
    test_toml_llm_quiver()
    test_no_env()
