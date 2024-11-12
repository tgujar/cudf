# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class GetJsonObjectOptions:
    def __init__(
        self,
        *,
        allow_single_quotes: bool = False,
        strip_quotes_from_single_strings: bool = True,
        missing_fields_as_nulls: bool = False,
    ) -> None: ...
    def get_allow_single_quotes(self) -> bool: ...
    def get_strip_quotes_from_single_strings(self) -> bool: ...
    def get_missing_fields_as_nulls(self) -> bool: ...
    def set_allow_single_quotes(self, val: bool) -> None: ...
    def set_strip_quotes_from_single_strings(self, val: bool) -> None: ...
    def set_missing_fields_as_nulls(self, val: bool) -> None: ...

def get_json_object(
    col: Column, json_path: Scalar, options: GetJsonObjectOptions | None = None
) -> Column: ...
