from typing import Annotated, Literal
import numpy as np
import numpy.typing as npt

Array = Annotated[npt.NDArray[np.float32], Literal["N"]]
"""The array type."""