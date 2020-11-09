import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS_lib.FQCS.manager import FQCSManager
from FQCS_lib.FQCS import fqcs_constants, fqcs_api, helper
import os
import asyncio
import os

API_URL = "http://localhost:60873"


async def main():
    res, data = fqcs_api.login(API_URL, "admin", "123123")
    token = data['access_token']
    print(data)
    headers = fqcs_api.get_common_headers(token)
    result, data = fqcs_api.count_events(API_URL,
                                         query_obj={"page": 1},
                                         headers=headers)
    print(result, data)
    return


if __name__ == "__main__":
    asyncio.run(main())