# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
                     
def load_env():
    _ = load_dotenv(find_dotenv(), override=True)

def get_phoenix_endpoint():
    load_env()
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    return phoenix_endpoint


