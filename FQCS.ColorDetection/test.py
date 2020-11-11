import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS_lib.FQCS import fqcs_constants, fqcs_api, helper
import os
import trio
import os

API_URL = "http://localhost:60873"


async def main():
    res, data = fqcs_api.login(API_URL, "admin", "123123")
    token = data['access_token']
    print(data)
    headers = fqcs_api.get_common_headers(token)
    # result, data = fqcs_api.submit_event(
    #     API_URL, [fqcs_constants.SIZE_MISMATCH, fqcs_constants.STAIN],
    #     'dirt.jpg',
    #     'dirt.jpg', ['dirt.jpg', 'dirt.jpg'],
    #     headers=headers)
    result, data = fqcs_api.submit_event(API_URL, [],
                                         'dirt.jpg',
                                         'dirt.jpg', ['dirt.jpg', 'dirt.jpg'],
                                         headers=headers)
    print(result, data)
    return


if __name__ == "__main__":
    trio.run(main)