import yaml
import toml
import json
import os


def write_text(text, dst):
    with open(dst, 'w', encoding='utf-8') as file:
        file.write(text)


def read_text(dst):
    with open(dst, 'r') as file:
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


class JSONObj(object):
    def __init__(self, input_dict):
        self._load_dict(input_dict)

    def _load_dict(self, input_dict):
        for key, value in input_dict.items():
            if key.startswith('#'):
                continue
            if isinstance(value, (list, tuple)):
                setattr(self, key, [JSONObj(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, JSONObj(value) if isinstance(value, dict) else value)

    def merge_dict(self, new_dict):
        for key, value in new_dict.items():
            if key.startswith('#'):
                continue
            if hasattr(self, key):
                if isinstance(value, dict):
                    getattr(self, key).merge_dict(value)
                else:
                    setattr(self, key, value)
            else:
                self._load_dict({key: value})


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, '_ast'):
        return todict(obj._ast())
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, '__dict__'):
        data = dict([
            (key, todict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, '__class__'):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def md5sum(file_path):
    import subprocess
    result = subprocess.run(['md5sum', file_path], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').split()[0]


def calculate_sha1(file_path):
    import hashlib
    sha1 = hashlib.sha1()
    try:
        with open(file_path, 'rb') as file:
            while True:
                data = file.read(8192)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()
    except FileNotFoundError:
        return "File not found."
