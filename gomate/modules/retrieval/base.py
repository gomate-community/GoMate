#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: retrievers.py
@time: 2024/05/30
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from abc import ABC, abstractmethod
from typing import List
import json
import os


class BaseRetriever(ABC):
    """通用的检索器接口"""

    def save_index(self):
        raise NotImplementedError
    def load_index(self):
        raise NotImplementedError
    def build_from_texts(self, corpus):
        """构建索引"""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query, top_k):
        """检索并返回前K个结果"""
        raise NotImplementedError



class BaseConfig:
    """
    Base configuration class that provides common methods for managing configurations.

    This class can be inherited by specific configuration classes (e.g., BM25RetrieverConfig, DenseRetrieverConfig)
    to implement shared methods like saving to a file, loading from a file, and logging the configuration.
    """

    def log_config(self):
        """Return a formatted string that summarizes the configuration."""
        config_summary = f"{self.__class__.__name__} Configuration:\n"
        for key, value in self.__dict__.items():
            config_summary += f"{key}: {value}\n"
        return config_summary

    def save_to_file(self, file_path):
        """Save the configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path):
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file {file_path} does not exist.")

        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def validate(self):
        """Validate configuration parameters. Override in subclasses if needed."""
        raise NotImplementedError("This method should be implemented in the subclass.")
