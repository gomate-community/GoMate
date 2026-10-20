import os
from copy import deepcopy
import yaml
from typing import Dict, Any, Optional
from loguru import logger

# Default configuration
DEFAULT_CONFIG = {
    "api": {
        "port": 3051
    },
    "openai": {
        "api_key": None,
        "base_url": "https://api.openai.com/v1",
        "model": "o3-mini"
    },
    "serper": {
        "api_key": None,
        "base_url": "https://google.serper.dev/search"
    },
    "tavily": {
        "api_key": None,
        "base_url": "https://api.tavily.com/search"
    },
    "research": {
        "default_breadth": 3,
        "default_depth": 2,
        "concurrency_limit": 2,
        "context_size": 128000,
        "search_source": "auto",
        "max_results_per_query": 5
    }
}

# Global configuration object
_config = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to environment variables
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing configuration
    """
    global _config

    # Start with default configuration
    config = deepcopy(DEFAULT_CONFIG)

    # Try to load from specified config file
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Merge with defaults (deep merge would be better but this is simple)
                    for section, values in yaml_config.items():
                        if section in config and isinstance(values, dict):
                            config[section].update(values)
                        else:
                            config[section] = values
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {str(e)}")
    else:
        # If no config file specified, look for config.yaml in the current directory
        default_path = './config.yaml'
        if os.path.exists(default_path):
            return load_config(default_path)
        else:
            logger.info("No configuration file found, using defaults")

    # Override with environment variables (for backward compatibility)
    # OpenAI
    if os.getenv("OPENAI_KEY"):
        config["openai"]["api_key"] = os.getenv("OPENAI_KEY")
    if os.getenv("OPENAI_ENDPOINT"):
        config["openai"]["base_url"] = os.getenv("OPENAI_ENDPOINT")
    if os.getenv("CUSTOM_MODEL"):
        config["openai"]["model"] = os.getenv("CUSTOM_MODEL")

    # Fireworks
    if os.getenv("FIREWORKS_KEY"):
        config["fireworks"]["api_key"] = os.getenv("FIREWORKS_KEY")

    # Serper
    if os.getenv("SERPER_KEY"):
        config["serper"]["api_key"] = os.getenv("SERPER_KEY")
    if os.getenv("SERPER_API_KEY"):
        config["serper"]["api_key"] = os.getenv("SERPER_API_KEY")

    # Tavily
    if os.getenv("TAVILY_KEY"):
        # Store the raw API key/token without the 'Bearer' prefix
        config["tavily"]["api_key"] = os.getenv("TAVILY_KEY")
    if os.getenv("TAVILY_BASE_URL"):
        config["tavily"]["base_url"] = os.getenv("TAVILY_BASE_URL")

    # Research settings
    if os.getenv("CONTEXT_SIZE"):
        try:
            config["research"]["context_size"] = int(os.getenv("CONTEXT_SIZE"))
        except (ValueError, TypeError):
            pass

    # Store the config globally
    _config = config
    return config


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration
    
    Returns:
        Dict containing configuration
    """
    global _config
    if _config is None:
        return load_config()
    # logger.debug(_config)
    return _config
