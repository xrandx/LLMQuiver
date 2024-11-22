import re
REGEX = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]{0,29})\}\}")


def extract_variable_names(template):
    return re.findall(REGEX, template)


def remove_template_variables(text: str):
    return re.sub(REGEX, r"{\1}", text)
