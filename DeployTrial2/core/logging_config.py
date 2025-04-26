# filepath: DeployTrial2/core/logging_config.py
# Optional: Centralized logging configuration if needed.
# For now, basic config is in main.py
import logging
from core.config import settings

# Example of more advanced configuration (e.g., multiple handlers, filters)
# def setup_logging():
#     logging.basicConfig(level=settings.LOG_LEVEL.upper(),
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # Add file handler, rotating handler, etc.
#     # file_handler = logging.FileHandler("app.log")
#     # file_handler.setLevel(logging.DEBUG)
#     # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # file_handler.setFormatter(formatter)
#     # logging.getLogger().addHandler(file_handler)
#     logging.getLogger(__name__).info("Advanced logging configured.")

# if __name__ == "core.logging_config":
#      setup_logging()
