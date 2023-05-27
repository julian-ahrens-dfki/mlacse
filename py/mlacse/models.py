import numpy as np
import onnxruntime as ort


def load_model(filename):
    session = ort.InferenceSession(filename)
    def transform(inp):
        x = np.copy(inp)
        shape = np.shape(x)
        x = np.reshape(x, (-1, shape[-1]))
        x /= 100
        x = x.astype(np.float32)
        x, = session.run(None, {'input': x})
        x *= 100
        x = np.reshape(x, (*shape[:-1], x.shape[-1]))
        return x
    return transform


