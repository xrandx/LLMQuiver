import yaml
import toml
import json
import os


def write_text(text, dst):
    with open(dst, 'w', encoding='utf-8') as file:
        file.write(text)


def read_text(src):
    with open(src, 'r') as file:
        return file.read()


def read_jsonl(src):
    with open(src, 'r') as json_file:
        res = []
        json_list = list(json_file)
        for json_str in json_list:
            res.append(json.loads(json_str))
        return res


def write_jsonl(obj, dst):
    with open(dst, 'w') as json_file:
        res = []
        for json_str in obj:
            res.append(json.dumps(json_str, ensure_ascii=False))
        json_file.write("\n".join(res))


def write_json(obj, dst):
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(src):
    with open(src, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def read_yaml(src):
    with open(src, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)


def convert_yaml_to_obj(yaml_str):
    return yaml.safe_load(yaml_str)


def convert_obj_to_yaml(obj):
    return yaml.dump(obj, indent=2, allow_unicode=True, sort_keys=False)


def write_yaml(obj, dst):
    with open(dst, 'w', encoding='utf-8') as file:
        try:
            yaml.safe_dump(obj, file, allow_unicode=True, sort_keys=False)
        except yaml.YAMLError as exc:
            print(exc)


def read_toml(src):
    with open(src, 'r') as toml_file:
        data = toml.load(toml_file)
    return data


def write_toml(obj, dst):
    with open(obj, 'w') as toml_file:
        toml.dump(dst, toml_file)


def read_file_lst(src, endswith=None):
    filenames = os.listdir(src)
    if endswith:
        filenames = filter(lambda x: x.endswith(endswith), filenames)
    return list(map(lambda x: os.path.join(src, x), filenames))
