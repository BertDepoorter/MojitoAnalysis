import few
for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
    print(f" - Backend '{backend}': {"available" if few.has_backend(backend) else "unavailable"}")
