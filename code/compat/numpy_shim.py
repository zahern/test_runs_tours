"""
NumPy 2.0 Compatibility Shim
============================
Provides removed NumPy 1.x aliases that were removed in NumPy 2.0.
Must be imported BEFORE any other imports that might use these aliases.

Place this import at the VERY TOP of your entry scripts, before any other imports.
"""
import numpy as np
import sys

# NumPy 1.x aliases removed in NumPy 2.0
# Source: https://numpy.org/devdocs/release/2.0.0-notes.html#deprecations
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'object_'):
    np.object_ = object
if not hasattr(np, 'int_'):
    np.int_ = int
if not hasattr(np, 'float_'):
    np.float_ = float
if not hasattr(np, 'bool_'):
    np.bool_ = bool
if not hasattr(np, 'complex_'):
    np.complex_ = complex
if not hasattr(np, 'object_'):
    np.object_ = object
if not hasattr(np, 'int8'):
    np.int8 = np.int8
if not hasattr(np, 'int16'):
    np.int16 = np.int16
if not hasattr(np, 'int32'):
    np.int32 = np.int32
if not hasattr(np, 'int64'):
    np.int64 = np.int64
if not hasattr(np, 'uint8'):
    np.uint8 = np.uint8
if not hasattr(np, 'uint16'):
    np.uint16 = np.uint16
if not hasattr(np, 'uint32'):
    np.uint32 = np.uint32
if not hasattr(np, 'uint64'):
    np.uint64 = np.uint64
if not hasattr(np, 'float16'):
    np.float16 = np.float16
if not hasattr(np, 'float32'):
    np.float32 = np.float32
if not hasattr(np, 'float64'):
    np.float64 = np.float64
if not hasattr(np, 'complex64'):
    np.complex64 = np.complex64
if not hasattr(np, 'complex128'):
    np.complex128 = np.complex128
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int0'):
    np.int0 = int
if not hasattr(np, 'float0'):
    np.float0 = float
if not hasattr(np, 'complex0'):
    np.complex0 = complex
if not hasattr(np, 'bool0'):
    np.bool0 = bool
if not hasattr(np, 'void0'):
    np.void0 = np.void

# Additional aliases that some packages expect
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode'):
    np.unicode = np.str_

# Suppress the NumPy 2.0 deprecation warning for modules that still use old aliases
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np\\.(unicode_|object_|int_|float_|bool_|complex_|object_|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float16|float32|float64|complex64|complex128|bool8|int0|float0|complex0|bool0|void0|string_|unicode).*",
    category=DeprecationWarning
)

# Mark as initialized
_numpy_compat_initialized = True

# Log initialization (only once)
import sys
if not getattr(sys.modules[__name__], '_numpy_compat_warned', False):
    import warnings
    warnings.warn(
        "NumPy 2.0 compatibility shim loaded. "
        "Some packages may still be using deprecated NumPy 1.x aliases. "
        "Consider upgrading packages to NumPy 2.0 compatible versions.",
        UserWarning,
        stacklevel=2
    )
    sys.modules[__name__]._numpy_compat_warned = True