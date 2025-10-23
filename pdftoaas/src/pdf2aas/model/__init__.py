"""Classes to handle properties their definitions and classes inside the library."""

from .class_definition import ClassDefinition
from .property import Property
from .property_definition import PropertyDefinition, SimplePropertyDataType, ValueDefinitionKeyType

__all__ = [
    "ClassDefinition",
    "Property",
    "PropertyDefinition",
    "SimplePropertyDataType",
    "ValueDefinitionKeyType",
]
