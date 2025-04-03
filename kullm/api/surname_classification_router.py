"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""

from fastapi import APIRouter

from kullm.domain.commons import ErrorCode
from kullm.domain.surname_req_res import ClassifySurnameResponse, ClassifySurnameRequest
from kullm.service.surname_classification_service import SurnameClassificationService
from kullm.util import loggerutil, strutil

logger = loggerutil.get_logger(__name__)

router = APIRouter(tags=["Surname Classification"])

service = SurnameClassificationService()


@router.post("/classify/surname", response_model=ClassifySurnameResponse)
async def classify_surname(req: ClassifySurnameRequest = None):
    try:
        if req is None:
            return ClassifySurnameResponse(error=True, code=ErrorCode.BAD_REQUEST, message="Invalid request")
        if strutil.is_empty(req.query):
            return ClassifySurnameResponse(error=True, code=ErrorCode.BAD_REQUEST, message="Invalid input")

        res = service.predict(req)
        return res
    except Exception as e:
        logger.exception(f"Error: {e}")
        return ClassifySurnameResponse(error=True, code=ErrorCode.INTERNAL_ERROR, message=str(e))
