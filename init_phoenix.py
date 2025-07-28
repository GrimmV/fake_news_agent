
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

from helper import get_phoenix_endpoint


def init_phoenix(project_name):

    PROJECT_NAME = project_name

    PHOENIX_ENDPOINT = get_phoenix_endpoint()

    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint= PHOENIX_ENDPOINT + "v1/traces",
        
    )

    OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)

    tracer = tracer_provider.get_tracer(__name__)
    
    return tracer