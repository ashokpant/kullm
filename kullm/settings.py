"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""
import os


class Settings:
    SURNAME_MODEL_FILE = os.getenv("SURNAME_MODEL_FILE", "./models/surname-model.bin")
    KU_LLM_API_HOST = os.getenv("KU_LLM_API_HOST", "0.0.0.0")
    KU_LLM_API_PORT = int(os.getenv("KU_LLM_API_PORT", "8009"))
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = "DEBUG"
