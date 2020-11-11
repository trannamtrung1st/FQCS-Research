import requests
from .fqcs_constants import ISO_DATE_FORMAT, SERVER_ISO_DATE_FORMAT
import datetime


def refresh_token(api_url, r_token, headers=None):
    try:
        form_data = {}
        form_data['grant_type'] = 'refresh_token'
        form_data['refresh_token'] = r_token
        url = "{}/api/users/login".format(api_url)
        resp = requests.post(url, data=form_data, headers=headers)
        if (resp.status_code >= 200 and resp.status_code < 300):
            data = resp.json()
            return (True, data)
        else:
            return (False, resp)
    except Exception as ex:
        return (None, ex)


def login(api_url, username, password, headers=None):
    try:
        form_data = {}
        form_data['username'] = username
        form_data['password'] = password
        url = "{}/api/users/login".format(api_url)
        resp = requests.post(url, data=form_data, headers=headers)
        if (resp.status_code >= 200 and resp.status_code < 300):
            data = resp.json()
            return (True, data)
        else:
            return (False, resp)
    except Exception as ex:
        return (None, ex)


def count_events(api_url, query_obj=None, headers=None):
    try:
        url = "{}/api/qc-events/count".format(api_url)
        resp = requests.get(url, query_obj, headers=headers)
        if (resp.status_code >= 200 and resp.status_code < 300):
            data = resp.json()
            return (True, data)
        else:
            return (False, resp)
    except Exception as ex:
        return (None, ex)


def submit_event(api_url,
                 defect_types,
                 left_img,
                 right_img,
                 side_images,
                 headers=None):
    try:
        data = {}
        if defect_types is not None:
            details = []
            for defect in defect_types:
                details.append({'defect_type_code': defect})
            data['details'] = details
        data['left_image'] = left_img
        data['right_image'] = right_img
        data['side_images'] = side_images
        data['date_format'] = SERVER_ISO_DATE_FORMAT
        utc_now = datetime.datetime.utcnow()
        utc_str = utc_now.strftime(ISO_DATE_FORMAT)
        data['created_time_str'] = utc_str
        url = "{}/api/qc-events".format(api_url)
        resp = requests.post(url, json=data, headers=headers)
        if (resp.status_code >= 200 and resp.status_code < 300):
            data = resp.json()
            return (True, data)
        else:
            return (False, resp)
    except Exception as ex:
        return (None, ex)


def get_common_headers(auth_token=None):
    return {"Authorization": f"Bearer {auth_token}"}
