import pickle
import json

def convert(obj):
    """Helper function to convert non-serializable objects."""
    try:
        json.dumps(obj)  # check if serializable
        return obj
    except TypeError:
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj)

with open('/home/ltoschi/Documents/code/wtw/log.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert to a JSON-serializable format
log_0 = convert(data['hardware_closed_loop'][0])
log_1 = convert(data['hardware_closed_loop'][1])

with open('/home/ltoschi/Documents/code/wtw/log_0.json', 'w') as f:
    json.dump(log_0, f, indent=4)

with open('/home/ltoschi/Documents/code/wtw/log_1.json', 'w') as f:
    json.dump(log_1, f, indent=4)
