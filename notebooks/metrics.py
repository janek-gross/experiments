import json
import basyx
from basyx.aas.adapter import aasx
from typing import Dict, Any, Tuple, List
from basyx.aas.model.submodel import SubmodelElementCollection, Property, MultiLanguageProperty, File, Range
from basyx.aas import model
from openai import OpenAI
import os
import numpy as np
import sqlite3
import re
import uuid
from pdf2aas.dictionary.core import Dictionary
from decimal import Decimal
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rapidfuzz import fuzz

class CustomDictionary(Dictionary):
    supported_releases = ['0.0']
    def get_class_url(self, class_id: str) -> str | None:
        """Get the web URL for the class of the class_id for details."""
        return None

    def get_property_url(self, property_id: str) -> str | None:
        """Get the web URL for the property id for details."""
        return None

def company_from_product_id(product_id):
    company = product_id.split("_")[0]
    return company

def normalize_id_short(id_short):
        anti_alphanumeric_regex = re.compile(r"[^a-zA-Z0-9]")
        id_short = re.sub(anti_alphanumeric_regex, "_", id_short)
        if len(id_short) == 0:
            key = ""
        elif not id_short[0].isalpha():
            id_short = "ID_" + id_short
        return id_short[:128]

def extract_properties(elements):
    collected = []
    for elem in elements:
        if isinstance(elem, model.submodel.SubmodelElementCollection):
            # Combine results from nested collections
            collected.extend(extract_properties(elem))
        elif isinstance(elem, (model.submodel.MultiLanguageProperty, model.submodel.Property)):
            collected.append(elem)
        else:
            print("Unknown Element", elem)
    return collected

def get_semantic_id(prop):
    if prop.semantic_id is not None:
        return prop.semantic_id.key[0].value
    else:
        return prop.id_short


def deduplicate_items_by_keys(items, keys_with_duplicates):
    if len(items) != len(keys_with_duplicates):
        raise ValueError("Items and keys_with_duplicates must have the same length.")

    seen = set()
    seen_idshort = set()
    result = []
    
    for item, key in zip(items, keys_with_duplicates):
        if key is None:
            # Remove only a single "_<number>" suffix if present
            base_name = re.sub(r'_[0-9]+$', '', getattr(item, "id_short", ""))
            if base_name not in seen_idshort:
                result.append(item)
                seen_idshort.add(base_name)
        else:
            if key not in seen:
                result.append(item)
                seen.add(key)
    return result

def get_value(value):
    if isinstance(value, model.MultiLanguageTextType):
        value = value.get('en', None) 
    elif isinstance(value, MultiLanguageProperty):
        value = value.value.get('en', None)
    elif isinstance(value, Range):
        value = f"{value.min} - {value.max}"
    if isinstance(value, Decimal):
        value = float(value)
    if value == "":
        value = None
    return str(value)
    
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# def extract_number(s):
#     match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
#     if match:
#         return float(match.group())
#     return None

def relative_difference(value, reference):
    # Relative difference capped at 1.0
    if abs(reference) < 1e-8:
        if abs(value) < 1e-8:
            return 0.0  # Both zero, no difference
        else:
            return 1.0  # Undefined or maximal difference
    
    elif (value < 0 and reference > 0) or (value > 0 and reference < 0):
        return 1.0
    
    # Normal case
    return min(abs(value / reference - 1),1)


class SubmodelComparison:
    def __init__(self, generated_submodels, db_path="embedding_cache.sqlite",
                 temp_dir="/app/data/processed/sample/"):
        self.generated_submodel_paths = generated_submodels
        self.db_path = db_path
        self.product_path = "/".join(generated_submodels[0].split('/')[:-1])
        self.product_id = generated_submodels[0].split('/')[-2] # assuming a folder for each product
        self.company = company_from_product_id(self.product_id)
        # self.reference_submodel_path = os.path.join(self.product_path, f'{self.product_id}_{self.company}_technical_data.json')
        self.reference_aasx_path = os.path.join(self.product_path, f'{self.product_id}.aasx')
        self.dictionary = CustomDictionary(release='0.0', temp_dir=temp_dir)
        self.dictionary.load_from_file(temp_dir+"CustomDictionary-0.0.json")
        self.class_properties = self.dictionary.get_class_properties(self.product_id)
        self.high_thresh = 0.80
        self.medium_thresh = 0.60
        self.tolerance = 1e-2

    def load_submodels(self):
        object_store = model.DictObjectStore()
        file_store = aasx.DictSupplementaryFileContainer()    
        with aasx.AASXReader(self.reference_aasx_path) as reader:
            # Read all contained AAS objects and all referenced auxiliary files
            reader.read_into(object_store=object_store,
                            file_store=file_store)
        for i in object_store:
            if i.id_short == "TechnicalData":
                technical_data_url = i.id
                break
        else:
            raise ValueError(f"TechnicalData submodel not found for {product_id}")
        self.reference_aasx = object_store
        self.reference_submodel = object_store.get_identifiable(technical_data_url)


        self.generated_submodels = []
        for path in self.generated_submodel_paths:
            with open(path, 'r') as gen_file:
                self.generated_submodels.append(json.load(gen_file,cls=basyx.aas.adapter.json.AASFromJsonDecoder))

    def _deduplicate_properties(self):
        """
        Deduplicates properties in the reference and generated submodels.
        Returns a tuple with the deduplicated properties for reference and generated submodels.
        """
        # Remove properties with empty value
        empty_values = [None, '']
        empty_values_semantic_ids = [get_semantic_id(prop) for prop in self.reference_properties if get_value(prop.value) in empty_values]
        self.reference_properties = [prop for prop in self.reference_properties
                                        if not any(get_value(prop.value) == sem_id for sem_id in empty_values_semantic_ids)
                                    ]
        self.class_properties =     [prop for prop in self.class_properties
                                        if not any(prop.id == sem_id for sem_id in empty_values_semantic_ids)
                                    ]



        class_semantic_ids = [i.id for i in self.class_properties]
        self.class_properties = deduplicate_items_by_keys(self.class_properties, class_semantic_ids)

        # Deduplicate reference properties
        reference_semantic_ids = [get_semantic_id(prop) for prop in self.reference_properties]

        # Remove properties that weren't part of the llm input
        for i in range(len(self.reference_properties) - 1, -1, -1):
            if reference_semantic_ids[i] not in class_semantic_ids:
                self.reference_properties.pop(i)
                reference_semantic_ids.pop(i)

        self.reference_properties = deduplicate_items_by_keys(self.reference_properties, reference_semantic_ids)

        # Deduplicate generated properties
        for i,props in enumerate(self.generated_properties):
            gen_semantic_ids = []
            for prop in props:
                if prop.semantic_id is not None:
                    gen_semantic_ids.append(prop.semantic_id.key[0].value)
                else:
                    gen_semantic_ids.append(None)
            self.generated_properties[i] = deduplicate_items_by_keys(props, gen_semantic_ids)


    def extract_properties(self):
        """
        Extracts properties from the reference submodel and all generated submodels.
        Returns a dictionary with the properties for each submodel.
        """
        self.reference_properties = extract_properties(
            self.reference_submodel.get_referable("TechnicalProperties"))


        # Extract properties from each generated submodel
        self.generated_properties = []
        for submodel in self.generated_submodels:
            props = extract_properties(submodel.get_referable("TechnicalProperties"))
            self.generated_properties.append(props)
        self._deduplicate_properties()

        id_short_dict = {prop.id: normalize_id_short(prop.name.get('en')) for prop in self.class_properties}
        values_dict = {get_semantic_id(prop): prop.value for prop in self.reference_properties}
        self.reference_property_names = [{'semantic_id': key, 'ref_key': id_short_dict[key], 'ref_value': get_value(values_dict[key])} for key in id_short_dict]

        self.generated_property_names = []
        for props in self.generated_properties:
            self.generated_property_names.append([{
                'semantic_id': prop.semantic_id.key[0].value if prop.semantic_id else None,
                'gen_key': prop.id_short,
                'gen_value': get_value(prop.value)
            } for prop in props])



    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
    def _load_embeddings_from_db(self, texts):
        cached = {}
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" for _ in texts)
            query = f"SELECT text, embedding FROM embeddings WHERE text IN ({placeholders})"
            for text, blob in conn.execute(query, texts):
                cached[text] = np.frombuffer(blob, dtype=np.float32)
        return cached

    def _save_embeddings_to_db(self, new_data):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO embeddings (text, embedding) VALUES (?, ?)",
                [(text, emb.astype(np.float32).tobytes()) for text, emb in new_data.items()]
            )

    def property_embeddings(self, model: str = "text-embedding-3-small"):
        reference_keys = [prop['ref_key'] for prop in self.reference_property_names]
        reference_values = [prop['ref_value'] for prop in self.reference_property_names]
        generated_keys = [[prop['gen_key'] for prop in gen_props] for gen_props in self.generated_property_names]
        generated_values = [[prop['gen_value'] for prop in gen_props] for gen_props in self.generated_property_names]
        texts = reference_keys + reference_values + \
        [key for keys in generated_keys for key in keys] + [value for values in generated_values for value in values]
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._init_db()
        embedding_cache = self._load_embeddings_from_db(texts)
        uncached_texts = [text for text in texts if text not in embedding_cache]
        embedding_batch_size = 200
        for i in range(0, len(uncached_texts), embedding_batch_size):
            batch = uncached_texts[i:i + embedding_batch_size]
            print(f"Verarbeite Batch {i // embedding_batch_size + 1} mit {len(batch)} Texten.")
            
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            new_embeddings = {
                text: np.array(d.embedding, dtype=np.float32)
                for text, d in zip(batch, response.data)
            }
            self._save_embeddings_to_db(new_embeddings)
            embedding_cache.update(new_embeddings)


        embeddings = [embedding_cache[text] for text in texts]
        # Split the embeddings into reference and generated
        self.reference_key_embeddings = [embedding_cache[text] for text in reference_keys]
        self.reference_value_embeddings = [embedding_cache[text] for text in reference_values]
        generated_key_embeddings = []
        for keys in generated_keys:
            generated_key_embeddings.append([embedding_cache[text] for text in keys])
        generated_value_embeddings = []
        for values in generated_values:
            generated_value_embeddings.append([embedding_cache[text] for text in values])
        self.generated_key_embeddings = generated_key_embeddings
        self.generated_value_embeddings = generated_value_embeddings

    def compute_similarity(self):
        # find and exclude exact matches
        self.matched_properties_semantic_id = []

        self.cost_matrices_embed = []
        self.cost_matrices_string_ratio = []
        self.cost_matrices_string_partial = []
        self.cost_matrices_string_token = []
        self.unmatched_generated = []
        self.matched_generated = []
        self.unmatched_reference = []
        self.matched_reference = []

        ref_prop_names = pd.DataFrame(self.reference_property_names)
        for i, gen_prop_names in enumerate(self.generated_property_names):
            gen_prop_names = pd.DataFrame(gen_prop_names, columns = ['semantic_id', 'gen_key', 'gen_value'])
            gen_ind = gen_prop_names[~gen_prop_names['semantic_id'].isnull()].index.to_list()
            ref_ind = ref_prop_names[ref_prop_names['semantic_id'].isin(gen_prop_names['semantic_id'])].index.to_list()
            sem_ids = ref_prop_names[ref_prop_names['semantic_id'].isin(gen_prop_names['semantic_id'])]['semantic_id'].to_list()
            self.matched_properties_semantic_id.append([{'semantic_id':row[0], 'reference_index':row[1], 'generated_index':row[2], 'similarity_key': 1.0} for row in zip(sem_ids,ref_ind,gen_ind)])

            self.unmatched_generated.append(gen_prop_names[gen_prop_names['semantic_id'].isnull()])
            self.unmatched_reference.append(ref_prop_names[~ref_prop_names['semantic_id'].isin(sem_ids)])
            ref_embeddings = np.array(self.reference_key_embeddings)[self.unmatched_reference[i].index.to_list()]
            gen_embeddings = np.array(self.generated_key_embeddings[i])[self.unmatched_generated[i].index.to_list()]
            gen_id_shorts = self.unmatched_generated[i]['gen_key'].values
            ref_id_shorts = self.unmatched_reference[i]['ref_key'].values

            cost_matrix_embed = np.zeros((len(ref_embeddings), len(gen_embeddings)))
            for i, ref_emb in enumerate(ref_embeddings):
                for j, gen_emb in enumerate(gen_embeddings):
                    sim = np.dot(ref_emb, gen_emb)
                    cost = 1 - sim
                    cost_matrix_embed[i][j] = cost
            self.cost_matrices_embed.append(cost_matrix_embed)

            cost_matrix_string = np.zeros((len(ref_id_shorts), len(gen_id_shorts)))
            for i, ref_id_short in enumerate(ref_id_shorts):
                for j, gen_id_short in enumerate(gen_id_shorts):
                    sim = fuzz.ratio(ref_id_short,gen_id_short)/100
                    cost = 1 - sim
                    cost_matrix_string[i][j] = cost
            self.cost_matrices_string_ratio.append(cost_matrix_string)
            cost_matrix_string = np.zeros((len(ref_id_shorts), len(gen_id_shorts)))
            for i, ref_id_short in enumerate(ref_id_shorts):
                for j, gen_id_short in enumerate(gen_id_shorts):
                    sim = fuzz.partial_ratio(ref_id_short,gen_id_short)/100
                    cost = 1 - sim
                    cost_matrix_string[i][j] = cost
            self.cost_matrices_string_partial.append(cost_matrix_string)
            cost_matrix_string = np.zeros((len(ref_id_shorts), len(gen_id_shorts)))
            for i, ref_id_short in enumerate(ref_id_shorts):
                for j, gen_id_short in enumerate(gen_id_shorts):
                    sim = fuzz.token_sort_ratio(ref_id_short.replace('_',' '),gen_id_short.replace('_',' '))/100
                    cost = 1 - sim
                    cost_matrix_string[i][j] = cost
            self.cost_matrices_string_token.append(cost_matrix_string)



    def match_properties_embed(self):
        self.matched_properties_embed = [] # List of match lists (per generated submodel)
        for i, cost_matrix in enumerate(self.cost_matrices_embed):
            semantic_ids = self.unmatched_reference[i]['semantic_id'].to_list()
            reference_index = self.unmatched_reference[i].index.to_list()
            generated_index = self.unmatched_generated[i].index.to_list()

            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for j, k in zip(row_ind, col_ind):
                score = 1 - cost_matrix[j, k]
                match = {
                    "semantic_id": semantic_ids[j],
                    "reference_index": reference_index[j],
                    "generated_index": generated_index[k],
                    "similarity_key": float(score),
                }
                matches.append(match)
            self.matched_properties_embed.append(matches)

        self.matched_properties_string_ratio = [] # List of match lists (per generated submodel)
        for i, cost_matrix in enumerate(self.cost_matrices_string_ratio):
            semantic_ids = self.unmatched_reference[i]['semantic_id'].to_list()
            reference_index = self.unmatched_reference[i].index.to_list()
            generated_index = self.unmatched_generated[i].index.to_list()

            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for j, k in zip(row_ind, col_ind):
                score = 1 - cost_matrix[j, k]
                match = {
                    "semantic_id": semantic_ids[j],
                    "reference_index": reference_index[j],
                    "generated_index": generated_index[k],
                    "similarity_key": float(score),
                }
                matches.append(match)
            self.matched_properties_string_ratio.append(matches)

        self.matched_properties_string_partial = [] # List of match lists (per generated submodel)
        for i, cost_matrix in enumerate(self.cost_matrices_string_partial):
            semantic_ids = self.unmatched_reference[i]['semantic_id'].to_list()
            reference_index = self.unmatched_reference[i].index.to_list()
            generated_index = self.unmatched_generated[i].index.to_list()

            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for j, k in zip(row_ind, col_ind):
                score = 1 - cost_matrix[j, k]
                match = {
                    "semantic_id": semantic_ids[j],
                    "reference_index": reference_index[j],
                    "generated_index": generated_index[k],
                    "similarity_key": float(score),
                }
                matches.append(match)
            self.matched_properties_string_partial.append(matches)

        self.matched_properties_string_token = [] # List of match lists (per generated submodel)
        for i, cost_matrix in enumerate(self.cost_matrices_string_token):
            semantic_ids = self.unmatched_reference[i]['semantic_id'].to_list()
            reference_index = self.unmatched_reference[i].index.to_list()
            generated_index = self.unmatched_generated[i].index.to_list()

            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for j, k in zip(row_ind, col_ind):
                score = 1 - cost_matrix[j, k]
                match = {
                    "semantic_id": semantic_ids[j],
                    "reference_index": reference_index[j],
                    "generated_index": generated_index[k],
                    "similarity_key": float(score),
                }
                matches.append(match)
            self.matched_properties_string_token.append(matches)




    def compute_similarity_values(self):
        for i,matches in enumerate(self.matched_properties_semantic_id):
            for j,match in enumerate(matches):
                similarity = float(np.dot(self.reference_value_embeddings[match['reference_index']],
                                    self.generated_value_embeddings[i][match['generated_index']]))
                match['similarity_value_cosine'] = similarity
                match['similarity_value_ratio'] = 1.0
                ref_value = self.reference_property_names[match['reference_index']]['ref_value']
                gen_value = self.generated_property_names[i][match['generated_index']]['gen_value']
                if is_float(ref_value):
                    if is_float(gen_value):
                        match['deviation'] = relative_difference(float(gen_value),float(ref_value))
                    else:
                        match['deviation'] = 1.0 # Max deviation for non-numeric
                else:
                    match['deviation'] = None


        for i,matches in enumerate(self.matched_properties_embed):
            for j,match in enumerate(matches):
                similarity = float(np.dot(self.reference_value_embeddings[match['reference_index']],
                                    self.generated_value_embeddings[i][match['generated_index']]))
                match['similarity_value_cosine'] = similarity
                ref_value = self.reference_property_names[match['reference_index']]['ref_value']
                gen_value = self.generated_property_names[i][match['generated_index']]['gen_value']
                if is_float(ref_value):
                    if is_float(gen_value):
                        match['deviation'] = relative_difference(float(gen_value),float(ref_value))
                    else:
                        match['deviation'] = 1.0 # Max deviation for non-numeric
                else:
                    match['deviation'] = None

        for i,matches in enumerate(self.matched_properties_string_ratio):
            for j,match in enumerate(matches):
                ref_value = self.reference_property_names[match['reference_index']]['ref_value']
                gen_value = self.generated_property_names[i][match['generated_index']]['gen_value']
                match['similarity_value_ratio'] = fuzz.ratio(ref_value,gen_value)/100
                if is_float(ref_value):
                    if is_float(gen_value):
                        match['deviation'] = relative_difference(float(gen_value),float(ref_value))
                    else:
                        match['deviation'] = 1.0 # Max deviation for non-numeric
                else:
                    match['deviation'] = None
        for i,matches in enumerate(self.matched_properties_string_partial):
            for j,match in enumerate(matches):
                ref_value = self.reference_property_names[match['reference_index']]['ref_value']
                gen_value = self.generated_property_names[i][match['generated_index']]['gen_value']
                match['similarity_value_partial'] = fuzz.partial_ratio(ref_value,gen_value)/100
                if is_float(ref_value):
                    if is_float(gen_value):
                        match['deviation'] = relative_difference(float(gen_value),float(ref_value))
                    else:
                        match['deviation'] = 1.0 # Max deviation for non-numeric
                else:
                    match['deviation'] = None
        for i,matches in enumerate(self.matched_properties_string_token):
            for j,match in enumerate(matches):
                ref_value = self.reference_property_names[match['reference_index']]['ref_value']
                gen_value = self.generated_property_names[i][match['generated_index']]['gen_value']
                match['similarity_value_ratio'] = fuzz.token_sort_ratio(ref_value.replace('_',' '),gen_value.replace('_',' '))/100
                if is_float(ref_value):
                    if is_float(gen_value):
                        match['deviation'] = relative_difference(float(gen_value),float(ref_value))
                    else:
                        match['deviation'] = 1.0 # Max deviation for non-numeric
                else:
                    match['deviation'] = None

    def _metrics(self, index):
        total_reference = len(self.reference_property_names)
        total_generated = len(self.generated_property_names[index])
        metrics = {}
        # Basic property counts
        metrics['total_reference_properties'] = total_reference  # Total number of properties in the original AAS
        metrics['total_generated_properties'] = total_generated  # Total number of properties in the generated (PDF-to-AAS) file

        # Raw property count comparison
        metrics['total_property_count_difference_raw'] = total_generated - total_reference  # Difference in raw property count (can be negative)
        metrics['total_property_count_abs_difference_raw'] = abs(total_generated - total_reference)  # Difference in raw property count (can be negative)

        # Match statistics
        metrics['one_to_one_matches'] = len(self.matched_properties_semantic_id[index])
        metrics['high_confidence_matches_embed'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] >= self.high_thresh])  # Properties confidently matched (above threshold)
        metrics['medium_confidence_matches_embed'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] >= self.medium_thresh])
        metrics['low_confidence_matches_embed'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] < self.medium_thresh])  # Properties with low-confidence matches (below threshold but considered related)
        metrics['mean_similarity_key_embed'] = float(np.mean([prop['similarity_key'] for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index]]))
        metrics['high_confidence_matches_ratio'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] >= self.high_thresh])  # Properties confidently matched (above threshold)
        metrics['medium_confidence_matches_ratio'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] >= self.medium_thresh])
        metrics['low_confidence_matches_ratio'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['similarity_key'] < self.medium_thresh])  # Properties with low-confidence matches (below threshold but considered related)
        metrics['mean_similarity_key_ratio'] = float(np.mean([prop['similarity_key'] for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index]]))



        # exact matching IE metrics
        metrics['key_recall_1to1'] = (metrics['one_to_one_matches']) / metrics['total_reference_properties']
        if metrics['total_generated_properties'] > 0:
            metrics['key_precision_1to1'] = (metrics['one_to_one_matches']) / metrics['total_generated_properties']
            if metrics['key_recall_1to1']*metrics['key_precision_1to1'] < 1e-8:
                metrics['key_f1_1to1'] = 0.0
            else:
                metrics['key_f1_1to1'] = 2*(metrics['key_recall_1to1']*metrics['key_precision_1to1'])/(metrics['key_recall_1to1'] + metrics['key_precision_1to1'])
        else:
            metrics['key_precision_1to1'] = 0.0
            metrics['key_f1_1to1'] = 0.0

        # embedding-based IE metrics
        metrics['key_recall_embed'] = (metrics['high_confidence_matches_embed']) / metrics['total_reference_properties']
        if metrics['total_generated_properties'] > 0:
            metrics['key_precision_embed'] = (metrics['high_confidence_matches_embed']) / metrics['total_generated_properties']
            if metrics['key_recall_embed']*metrics['key_precision_embed'] < 1e-8:
                metrics['key_f1_embed'] = 0.0
            else:
                metrics['key_f1_embed'] = 2*(metrics['key_recall_embed']*metrics['key_precision_embed'])/(metrics['key_recall_embed'] + metrics['key_precision_embed'])
        else:
            metrics['key_precision_embed'] = 0.0
            metrics['key_f1_embed'] = 0.0
        # levenstein-ratio-based IE metrics
        metrics['key_recall_ratio'] = (metrics['high_confidence_matches_ratio']) / metrics['total_reference_properties']
        if metrics['total_generated_properties'] > 0:
            metrics['key_precision_ratio'] = (metrics['high_confidence_matches_ratio']) / metrics['total_generated_properties']
            if metrics['key_recall_ratio']*metrics['key_precision_ratio'] < 1e-8:
                metrics['key_f1_ratio'] = 0.0
            else:
                metrics['key_f1_ratio'] = 2*(metrics['key_recall_ratio']*metrics['key_precision_ratio'])/(metrics['key_recall_ratio'] + metrics['key_precision_ratio'])
        else:
            metrics['key_precision_ratio'] = 0.0
            metrics['key_f1_ratio'] = 0.0

        metrics['total_reference_numeric'] = sum([is_float(prop['ref_value']) for prop in self.reference_property_names]) # Number of reference values convertable to float
        metrics['total_generated_numeric'] = sum([is_float(prop['gen_value']) for prop in self.generated_property_names[index]]) # Number of reference values convertable to float

        # Value statistics one-to-one matching
        metrics['mean_similarity_value_1to1'] = float(np.mean([prop['similarity_value_cosine'] for prop in self.matched_properties_semantic_id[index] if prop['deviation']==None]))
        metrics['total_values_matched_1to1_numeric'] = len([prop for prop in self.matched_properties_semantic_id[index] if prop['deviation'] != None])
        metrics['total_values_matched_1to1_string'] = len([prop for prop in self.matched_properties_semantic_id[index] if prop['deviation'] == None])

        metrics['value_matches_1to1_numeric'] = len([prop for prop in self.matched_properties_semantic_id[index] if prop['deviation']!=None and prop['deviation'] < self.tolerance])
        metrics['value_matches_string_1to1_high_conf'] = len([prop for prop in self.matched_properties_semantic_id[index] if prop['deviation']==None and prop['similarity_value_cosine'] >= self.high_thresh])

        metrics['value_recall_1to1'] = (metrics['value_matches_1to1_numeric'] + metrics['value_matches_string_1to1_high_conf'])/metrics['total_reference_properties']

        if metrics['total_generated_properties'] > 0:
            metrics['value_precision_1to1'] = (metrics['value_matches_1to1_numeric'] + metrics['value_matches_string_1to1_high_conf'])/metrics['total_generated_properties']
            if metrics['value_recall_1to1']*metrics['value_precision_1to1'] < 1e-8:
                metrics['value_f1_1to1'] = 0.0
            else:
                metrics['value_f1_1to1'] = 2*(metrics['value_recall_1to1']*metrics['value_precision_1to1'])/(metrics['value_recall_1to1']+metrics['value_precision_1to1'])
        else:
            metrics['value_precision_1to1'] = 0.0
            metrics['value_f1_1to1'] = 0.0


        # Value statistics embedding matching
        metrics['mean_similarity_value_embed'] = float(np.mean([prop['similarity_value_cosine'] for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['deviation']==None]))
        metrics['total_values_matched_embed_numeric'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['deviation'] != None])
        metrics['total_values_matched_embed_string'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['deviation'] == None])

        metrics['value_matches_embed_numeric'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['deviation']!=None and prop['deviation'] < self.tolerance])
        metrics['value_matches_string_embed_high_conf'] = len([prop for prop in self.matched_properties_embed[index]+self.matched_properties_semantic_id[index] if prop['deviation']==None and prop['similarity_value_cosine'] >= self.high_thresh])

        metrics['value_recall_embed'] = (metrics['value_matches_embed_numeric'] + metrics['value_matches_string_embed_high_conf'])/metrics['total_reference_properties']

        if metrics['total_generated_properties'] > 0:
            metrics['value_precision_embed'] = (metrics['value_matches_embed_numeric'] + metrics['value_matches_string_embed_high_conf'])/metrics['total_generated_properties']
            if metrics['value_recall_embed']*metrics['value_precision_embed'] < 1e-8:
                metrics['value_f1_embed'] = 0.0
            else:
                metrics['value_f1_embed'] = 2*(metrics['value_recall_embed']*metrics['value_precision_embed'])/(metrics['value_recall_embed']+metrics['value_precision_embed'])
        else:
            metrics['value_precision_embed'] = 0.0
            metrics['value_f1_embed'] = 0.0

        # Value statistics levenstein ratio matching
        metrics['mean_similarity_value_ratio'] = float(np.mean([prop['similarity_value_ratio'] for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['deviation']==None]))
        metrics['total_values_matched_ratio_numeric'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['deviation'] != None])
        metrics['total_values_matched_ratio_string'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['deviation'] == None])

        metrics['value_matches_ratio_numeric'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['deviation']!=None and prop['deviation'] < self.tolerance])
        metrics['value_matches_string_ratio_high_conf'] = len([prop for prop in self.matched_properties_string_ratio[index]+self.matched_properties_semantic_id[index] if prop['deviation']==None and prop['similarity_value_ratio'] >= self.high_thresh])

        metrics['value_recall_ratio'] = (metrics['value_matches_ratio_numeric'] + metrics['value_matches_string_ratio_high_conf'])/metrics['total_reference_properties']

        if metrics['total_generated_properties'] > 0:
            metrics['value_precision_ratio'] = (metrics['value_matches_ratio_numeric'] + metrics['value_matches_string_ratio_high_conf'])/metrics['total_generated_properties']
            if metrics['value_recall_ratio']*metrics['value_precision_ratio'] < 1e-8:
                metrics['value_f1_ratio'] = 0.0
            else:
                metrics['value_f1_ratio'] = 2*(metrics['value_recall_ratio']*metrics['value_precision_ratio'])/(metrics['value_recall_ratio']+metrics['value_precision_ratio'])
        else:
            metrics['value_precision_ratio'] = 0.0
            metrics['value_f1_ratio'] = 0.0



        # # None or empty values
        # "total_generated_properties_none_value": none_count_generated,  # Number of generated properties with a None/empty value
        # "total_generated_properties_without_none_value": non_none_generated,  # Generated properties that actually contain data (excluding Nones)

        # # Effective difference (ignoring Nones)
        # "total_property_count_difference_effective": non_none_generated - total_original,  # Effective property count difference after excluding empty values

        # # Quality indicator for generated values
        # "generated_properties_none_value_ratio": none_count_generated / total_generated if total_generated else 0.0,  # Ratio of empty properties in the generated file
        # "generated_properties_non_empty_ratio": non_none_generated / total_generated if total_generated else 0.0,  # Ratio of non empty properties in the generated file
        

        
        
        # "high_conf_match_ratio_from_matches": high_conf_count / total_matches if total_matches else 0.0,
        # "medium_conf_match_ratio_from_matches": medium_conf_count / total_matches if total_matches else 0.0,
        # "low_conf_match_ratio_from_all": low_conf_count / (total_original if total_original else 1),
        

        # # Original unmatched stats
        # "unmatched_original_high_conf_only": total_original - high_conf_count,  # How many original properties couldn't be matched with high confidence
        # "unmatched_original_high_medium_conf": total_original - (high_conf_count + medium_conf_count),
        # "matched_original_ratio_high_conf": high_conf_count / total_original,
        # "unmatched_original_ratio": 1 - high_conf_count / total_original if total_original else 0.0,  # missrate: % of original properties that were missed

        # # Generated unmatched stats
        # "unmatched_generated_high_conf_only": total_generated - high_conf_count,  # How many original properties couldn't be matched with high confidence
        # "unmatched_generated_high_medium_conf": total_generated - (high_conf_count + medium_conf_count),
        # "unmatched_generated_without_none": non_none_generated - high_conf_count,  # Generated non-empty properties that were unmatched

        #"extra_generated_keys_ratio": unmatched_generated / total_generated,
        
        # # Coverage from generated side
        # "matched_generated_ratio_high_conf": high_conf_count / total_generated if total_generated else 0.0,  # % of generated properties confidently matched
        # "unmatched_generated_ratio": 1 - high_conf_count / total_generated if total_generated else 0.0,  # % of generated properties that were unmatched
        # "matched_non_empty_generated_ratio": high_conf_count / non_none_generated if non_none_generated else 0.0,


        # # Overall alignment indicator
        # "match_coverage_ratio": high_conf_count / (total_original + total_generated) if (total_original + total_generated) else 0.0,  # balanced coverage ratio: Matches relative to total number of properties across both submodels

        # # Null value quality
        # "generated_none_ratio": none_count_generated / total_generated if total_generated else 0.0,
        
        # "precision_high_conf": high_conf_count / total_generated if total_generated else 0.0,
        # "recall_high_conf":    high_conf_count / total_original if total_original else 0.0,
        # "f1_high_conf": (
        #     2 * high_conf_count / (total_generated + total_original)
        #     if (total_generated + total_original) else 0.0
        # ),

        return metrics


    def compute_metrics(self):
        self.matched_properties = [i+j for i,j in zip(self.matched_properties_semantic_id,self.matched_properties_embed)]
        self.results = [self._metrics(i) for i in range(len(self.matched_properties))]


    
    # def _count_none_values(self, props: Dict[str, Any]) -> int:
    #     count = 0
    #     for prop in props:
    #         val = prop.values()
    #         if isinstance(val, list):  # MultiLanguageProperty
    #             if all(
    #                 isinstance(entry, dict) and (
    #                     entry.get("text") is None or
    #                     str(entry.get("text")).strip().lower() == "none" or
    #                     str(entry.get("text")).strip() == ""
    #                 )
    #                 for entry in val
    #             ):
    #                 count += 1
    #         else:  # Regular Property
    #             if val is None or str(val).strip().lower() == "none" or str(val).strip() == "":
    #                 count += 1
    #     return count
    

    def run(self):
        self.load_submodels()
        self.extract_properties()
        self.property_embeddings()
        self.compute_similarity()
        self.match_properties_embed()
        self.compute_similarity_values()
        self.compute_metrics()
