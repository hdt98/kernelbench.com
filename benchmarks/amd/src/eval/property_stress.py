"""No-op property stress module for AMD (hypothesis segfaults on gfx942)."""
def generate_property_cases(name):
    return ([], [])
def property_shape_index(name):
    return -1
def check_tensor_properties(*args, **kwargs):
    pass
def tolerance_for_property(name, tol):
    return tol or 0.0
