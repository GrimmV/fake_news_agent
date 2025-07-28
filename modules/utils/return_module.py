from opentelemetry.trace import StatusCode

def return_module(function, params, datapoint_id: int = None, span = None):
    if datapoint_id:
        the_module = function(**params, dp_id=datapoint_id)
    else:
        the_module = function(**params)
        
    if span:
        span.set_input({
            "params": params,
            "function": function.__name__
        })
        span.set_output(the_module["raw"])
        span.set_status(StatusCode.OK)
            
    visual = the_module["visual"]
    visual.update_layout(
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return the_module