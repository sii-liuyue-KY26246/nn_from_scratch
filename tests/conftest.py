# tests/conftest.py
# pytest 会自动加载这个文件。
# 将项目根目录加入 sys.path，确保 import nn_from_scratch 能工作。

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
