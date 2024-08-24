from llm_quiver.llm_quiver import LLMQuiver
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path

# .env file
env_path = Path("tests/unit_tests/.test.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


# setup LLMQUIVER_CONFIG, API_BASE, API_KEY
llm = LLMQuiver(
    prompt_name="hello_world_template",
    prompt_toml_file="tests/unit_tests/prompts/hello_world.toml",
)
templ_vals = [dict(name="GPT")]
responses = llm.generate(templ_vals=templ_vals)
logger.info(f"responses: {responses}")
