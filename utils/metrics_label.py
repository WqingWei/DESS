import numpy as np
from .generics import convert_vector_to_events
from .metrics import pr_from_events
 
def affiliation_f(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return F