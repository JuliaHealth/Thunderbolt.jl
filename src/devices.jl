# KA compat
# TODO upstream
default_backend(::SequentialCPUDevice) = KA.CPU()
default_backend(::PolyesterDevice) = KA.CPU()
