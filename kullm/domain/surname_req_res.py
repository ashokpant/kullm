"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""
from kullm.domain.commons import BaseRequest, BaseResponse


class ClassifySurnameRequest(BaseRequest):
    query: str = None


class ClassifySurnameResponse(BaseResponse):
    category: str = None
