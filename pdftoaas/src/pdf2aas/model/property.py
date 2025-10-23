"""Classes and functions to work with extracted technical properties."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .property_definition import PropertyDefinition

_number_regex = r"([-+]?[0-9_]*\.?[0-9_]+)"
_numeric_range_regex = _number_regex + r".+?" + _number_regex


def try_cast_number(value: Any) -> float | int | None:
    """Try to cast the value to float and check if it is an integer."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    if value.is_integer():
        value = int(value)
    return value


@dataclass
class Property:
    """A dataclass to represent a property with a value that was extracted.

    Attributes:
        label (str): The label of the property, e.g. "Rated Torque".
        value (Any): The extracted value of the property.
        unit (str | None): The measurement unit for the given value.
        reference (str | None): A reference (~100 chars) where the value was found,
            e.g. an excerpt, or page reference.
        defintion (PropertyDefinition | None): Definition of the property if available.
        language(str): Language code (default en) used for the fields
            (except maybe reference, when it was translated).
        id (str): Optional ID to identify the property globally.

    """

    label: str = ""
    value: Any = None
    unit: str | None = None
    reference: str | None = None
    definition: PropertyDefinition | None = field(default=None, repr=False)
    language: str = field(default="en", repr=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False, compare=False)

    @property
    def definition_id(self) -> str | None:
        """Get the id of the definition if available, else None."""
        return self.definition.id if self.definition is not None else None

    @property
    def definition_name(self) -> str | None:
        """Get the definition name for the property language.

        Returns the first definition name if selected language is not available.
        Returns None if no definition or name is available.
        """
        if self.definition is None:
            return None
        return self.definition.get_name(self.language)

    def parse_numeric_range(self) -> tuple[float | int | None, float | int | None]:
        """Try to parse the value as a numerical range.

        Returns (None,None) if not parseable.
        Returns first and last argument if value is a collection (list, tuple, set, dict).
        """
        value = (self.value, self.value)
        if isinstance(self.value, list | tuple | set | dict):
            if len(self.value) == 0:
                return None, None
            value_list = (
                list(self.value.values()) if isinstance(self.value, dict) else list(self.value)
            )
            value = (value_list[0], value_list[-1])
        elif isinstance(self.value, str):
            result = re.search(_numeric_range_regex, self.value)
            if result is not None:
                value = (result.group(1), result.group(2))
        min_ = try_cast_number(value[0])
        max_ = try_cast_number(value[1])
        if min_ is not None and max_ is not None and min_ > max_:
            return (max_, min_)
        return (min_, max_)

    def to_legacy_dict(self) -> dict[str, str | None]:
        """Return dictionary format used before a Property class was defined.

        Contains the fields extracted from the document:
          - "property" with the label of the property,
          - "value" with the value of the property,
          - "unit"
          - "reference"
        And the fields added from the property definition:
          - "id" with the semantic id of the definition
          - "name" with the name from the defition

        """
        return {
            "property": self.label,
            "value": self.value,
            "unit": self.unit,
            "reference": self.reference,
            "id": self.definition.id if self.definition else "",
            "name": self.definition_name,
        }

    @staticmethod
    def from_dict(property_dict: dict) -> "Property":
        """Parse a Property from a dictionary."""
        label = property_dict.get("property")
        if label is None:
            label = property_dict.get("label")
        if label is None:
            label = ""

        return Property(
            label,
            property_dict.get("value"),
            property_dict.get("unit"),
            property_dict.get("reference"),
            None,
            property_dict.get("language", "en"),
        )
