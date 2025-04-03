import logging

# Configure logging to write to a file
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define hook functions
def log_kwargs(**kwargs):
    logging.info(f"Function called with kwargs: {kwargs}")

def log_exception(exception: Exception):
    logging.error(f"An exception occurred: {str(exception)}")