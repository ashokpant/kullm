"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 29/11/2024
"""
import os
import sys

from fastapi.middleware.cors import CORSMiddleware

from kullm.api import surname_classification_router, health_router
from kullm.settings import Settings
from kullm.util import loggerutil

sys.path.append(os.getcwd())

import uvicorn
import nest_asyncio

nest_asyncio.apply()

from fastapi import FastAPI

logger = loggerutil.get_logger(__name__)
app = FastAPI(
    title="KU LLM API",
    description="KU LLM API for different LLM use cases",
    version="1.0.0",
    author="Ashok Kumar Pant",
    email="ashok@treeleaf.ai",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router.router)
app.include_router(surname_classification_router.router, prefix="")

if __name__ == "__main__":
    loggerutil.setup_logging()
    uvicorn.run("main:app", host=Settings.KU_LLM_API_HOST, port=Settings.KU_LLM_API_PORT, reload=Settings.DEBUG,
                loop="asyncio", )
