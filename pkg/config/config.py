import os
import yaml

from dotenv import load_dotenv
load_dotenv()


CONFIG_FILE = os.environ.get("CONFIG_FILE", "resources/config.yml")
cfg = yaml.safe_load(open(file=CONFIG_FILE, encoding='utf-8'))
