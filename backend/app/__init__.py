# Size native thread pools (OpenMP/BLAS: numpy, scikit-learn, PyTorch) to the container's real
# CPU quota before any of those libraries load. Left alone, they start one thread per *host* core;
# inside a quota-limited container that oversubscription made a 0.1 s model call take minutes.
from app.resources import limit_threads as _limit_threads

_limit_threads()
