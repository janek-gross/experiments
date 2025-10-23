"""Extractors that search for property definitions directly."""

import logging
from typing import Literal

from openai import AzureOpenAI, OpenAI
from tabulate import tabulate

from pdf2aas.model import Property, PropertyDefinition

from . import CustomLLMClient, PropertyLLM

logger = logging.getLogger(__name__)


class PropertyLLMSearch(PropertyLLM):
    """PropertyLLM that searches for given property definitions.

    Attributes:
        system_prompt_template (str): String that instructs the LLM to search
            for the defined properties.
        use_property_definition (bool): Add the property definition text in the
            prompt.
        use_property_unit (bool): Add the desired unit of measurement in the
            prompt.
        use_property_values (bool): Add a list of allowed values in the prompt.
        use_property_datatype (bool): Add the property datatype in the prompt.
        max_definition_chars (int): Limit the lenght of the definition text to
            the amount of chars. 0 means no limit. Only active with
            `use_property_definition`.
        max_values_length (int): Limit the number values from the value list.
            0 means no limit. Only active with `use_property_values`.
        prompt_order (list[Literal["datasheet", "properties", "hint"]]):
            Determines order of parts of the search prompt. Used to optimize
            prompt caching, e.g. when the same properties are searches on
            multiple datasheets vs. when same datasheet is searched for
            different properties.
        property_table_format (str): output format for the property defintions.
            Defaults to "github" (markdown). See `tabulate.tabulate_formats` for
            available options. Examples are: 'github', 'html', 'simple', 'tsv'.

    """

    system_prompt_template = (
        PropertyLLM.system_prompt_template
        + """

Search exactly for the reuqested properties.
Keep the order of the requested properties.
The property field is used to assign your extracted property to the requested property definitions.
Convert the value to the requested unit if provided.
When multiple values apply use a json list to represent them.
Represent ranges as json list of two values.
"""
    )
    property_table_format: str = "github"

    def __init__(
        self,
        model_identifier: str,
        api_endpoint: str | None = None,
        client: OpenAI | AzureOpenAI | CustomLLMClient | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        property_keys_in_prompt: list[Literal["definition", "unit", "values", "datatype"]]
        | None = None,
        prompt_order: list[Literal["datasheet", "properties", "hint"]] | None = None,
    ) -> None:
        """Initialize the property LLM search with default values."""
        super().__init__(
            model_identifier,
            api_endpoint,
            client,
            temperature,
            max_tokens,
            response_format,
        )
        if property_keys_in_prompt is None:
            property_keys_in_prompt = []
        self.use_property_definition = "definition" in property_keys_in_prompt
        self.use_property_unit = "unit" in property_keys_in_prompt
        self.use_property_values = "values" in property_keys_in_prompt
        self.use_property_datatype = "datatype" in property_keys_in_prompt
        self.max_definition_chars = 0
        self.max_values_length = 0
        if prompt_order is None:
            prompt_order = ["datasheet", "properties", "hint"]
        self.prompt_order = prompt_order

    def create_prompt(
        self,
        datasheet: str,
        properties: PropertyDefinition | list[PropertyDefinition],
        language: str = "en",
        hint: str | None = None,
    ) -> str:
        """Create the prompt from the given datasheet and property definiions.

        Usefull to check the prompt upfront, customize or display it.

        The prompt is created based on the `prompt_order`:
        - The datasheet text is added with a small instruction.
        - Then the properties are added to the prompt. A PropertyDefinition is
          formated different (more descriptiv, c.f. :meth: create_property_prompt)
          than a list of PropertyDefinitions (c.f. :meth: create_property_list_prompt),
          even if only one definition is part of the list.
        - The `hint` (c.f. "prompt_hint" in :meth: extract) is optionally added to
          provide context or additional instructions.

        """
        prompt = ""
        for part in self.prompt_order:
            if part == "datasheet":
                prompt += f"The following text enclosed in triple backticks is the datasheet of the technical device. It was converted from pdf.\n```\n{datasheet}\n```\n"  # noqa: E501
            elif part == "properties":
                if isinstance(properties, list):
                    prompt += self.create_property_list_prompt(properties, language)
                else:
                    prompt += self.create_property_prompt(properties, language)
            elif part == "hint" and hint:
                prompt += hint + "\n"
        return prompt

    def create_property_prompt(
        self,
        property_: PropertyDefinition,
        language: str = "en",
    ) -> str:
        """Create the prompt to search for a given property.

        Formats it as descriptive text using the configured fields from the
        property defintion.
        """
        if len(property_.name) == 0:
            error = f"Property {property_.id} has no name."
            raise ValueError(error)
        property_name = property_.get_name(language)

        prompt = ""
        property_definition = property_.definition.get(language)
        if self.use_property_definition and property_definition:
            if (
                self.max_definition_chars > 0
                and len(property_definition) > self.max_definition_chars
            ):
                property_definition = property_definition[: self.max_definition_chars] + " ..."
            prompt += f'The "{property_name}" is defined as "{property_definition[:self.max_definition_chars]}".\n'  # noqa: E501
        if self.use_property_datatype and property_.type:
            prompt += f'The "{property_name}" has the datatype "{property_.type}".\n'
        if self.use_property_unit and property_.unit:
            prompt += f'The "{property_name}" has the unit of measure "{property_.unit}".\n'
        if self.use_property_values and len(property_.values) > 0:
            property_values = property_.values_list
            if self.max_values_length > 0 and len(property_values) > self.max_values_length:
                property_values = property_values[: self.max_values_length] + ["..."]
            prompt += f'The "{property_name}" can be one of these values: "{property_values}".\n'

        prompt += f'What is the "{property_name}" of the device?\n'
        return prompt

    def create_property_list_prompt(
        self,
        property_list: list[PropertyDefinition],
        language: str = "en",
    ) -> str:
        """Create the prompt to search for a list of property definitions.

        Formats them as a markdown table using the configured fields from the
        property definitions.

        Limits the definition and values according to the configured values.
        """
        headers = ["Property"]
        if self.use_property_datatype:
            headers.append("Datatype")
        if self.use_property_unit:
            headers.append("Unit")
        if self.use_property_definition:
            headers.append("Definition")
        if self.use_property_values:
            headers.append("Values")

        table_data = []
        for property_ in property_list:
            if len(property_.name) == 0:
                logger.warning("Property %s has no name.", property_.id)
                continue
            table_data.append(self._create_property_list_prompt_row(property_, language))

        return "Extract the following properties from the provided datasheet:\n" + tabulate(
            table_data, headers=headers, tablefmt=self.property_table_format,
        )

    def _create_property_list_prompt_row(
        self,
        property_: PropertyDefinition,
        language: str,
    ) -> list:
        property_name = property_.get_name(language)
        row = [property_name]
        if self.use_property_datatype:
            row.append(property_.type)
        if self.use_property_unit:
            row.append(property_.unit)
        if self.use_property_definition:
            property_definition = property_.get_definition(language)
            if (
                property_definition
                and self.max_definition_chars > 0
                and len(property_definition) > self.max_definition_chars
            ):
                property_definition = property_definition[: self.max_definition_chars] + " ..."
            row.append(property_definition)
        if self.use_property_values and len(property_.values) > 0:
            property_values = property_.values_list
            if self.max_values_length > 0 and len(property_values) > self.max_values_length:
                property_values = property_values[: self.max_values_length] + ["..."]
            row.append(str(property_values))
        return row

    def _add_definitions(
        self,
        properties: list[Property],
        property_definition: list[PropertyDefinition] | PropertyDefinition,
    ) -> list[Property]:
        if len(properties) == 0:
            return []
        if isinstance(property_definition, PropertyDefinition):
            property_definition = [property_definition]

        if len(properties) == len(property_definition):
            for i, property_ in enumerate(properties):
                property_.definition = property_definition[i]
        elif len(property_definition) == 1 and len(properties) > 1:
            logger.warning("Extracted %s properties for one definition.", len(properties))
            for property_ in properties:
                property_.definition = property_definition[0]
        else:
            logger.warning(
                "Extracted %s properties for %s definitions.",
                len(properties),
                len(property_definition),
            )
            property_definition_dict = {
                next(iter(p.name.values()), p.id).lower(): p for p in property_definition
            }
            for property_ in properties:
                property_.definition = property_definition_dict.get(property_.label.strip().lower())
        return properties
