"""Classes to represent articles that can be evaluated."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pdf2aas.model import PropertyDefinition

DictionaryNameType = Literal["ECLASS", "ETIM", "CDD"]

@dataclass
class EvaluationArticle:
    """Represents an article with extracted properties.

    Arguments:
        name (str): short name to identify the article, e.g. it's article number
        datasheet_path (str, optional): The file path to the article's datasheet.
        aasx_path (str, optional): The path to an aasx package of the article.
        definitions (list[PropertyDefinition]): Property definitions to be evaluated.
        values (dict[str, Any]): Mapping of definition ids to values to be checked against
          extracted values.
        class_ids(dict[dict[DictionaryNameType], dict[str,str]]): class ids
          in different dictionarys, in form: dictionary -> release -> id.
        datasheet_text (str, optional): The preprocessed datasheet text used for
          extraction.

    """

    name: str
    datasheet_path: str | None = None
    aasx_path: str | None = None
    definitions: list[PropertyDefinition] = field(default_factory=list)
    # TODO: replace with list[Property]?
    values: dict[str, Any] = field(default_factory=dict)
    class_ids: dict[DictionaryNameType, dict[str, str]] = field(
        default_factory=dict,
    )
    datasheet_text: str | None = None
    # TODO: add method to load from aasx file / AASTemplate
