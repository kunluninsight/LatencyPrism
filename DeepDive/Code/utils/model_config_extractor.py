# file: utils.model_config_extractor.py
# Copyright (c) 2026, Alibaba Cloud. All rights reserved.
"""
This module provides the functionality to extract key configuration information 
from model configuration log files.

Its primary purpose is to parse log content to retrieve essential parameters such
as model name, hardware configuration, parallelism strategy, and model precision.
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from io import StringIO

def _get_priority(value: Any) -> int:
    """
    Assigns a priority score to an extracted value.

    Since the same configuration item may be found in multiple places, this function
    helps decide which value is the most reliable. A higher score indicates more
    specific and trustworthy information.

    Priority Rules:
    - 4: Highly specific values (e.g., 'FP16', 'sm80', or any number).
    - 3: Regular, non-empty strings.
    - 2: Strings indicating auto-configuration or null values (e.g., 'AUTO', '(NULL)').
    - 1: Strings indicating unknown values (e.g., '未知').
    - 0: None values, which have the lowest priority.

    Args:
        value: The value to be scored.

    Returns:
        An integer representing the value's priority.
    """
    if value is None: return 0
    val_str = str(value).upper()
    if not val_str or val_str == "未知": return 1
    if "AUTO" in val_str or "(NULL)" in val_str: return 2
    if any(p in val_str for p in ['BF16', 'FP16', 'FP8', 'INT', 'SM', 'HOPPER', 'AMPERE']) or re.search(r'\d', val_str): return 4
    return 3

def _update_results_legacy(results: Dict, target_keys: Tuple[str, ...], value: Any):
    """
    Conditionally updates a value in a nested dictionary based on priority.

    This function navigates to a nested key within the `results` dictionary and
    updates the value only if the new `value` has a priority greater than or equal
    to the existing one. This ensures that more specific information always
    overwrites generic or default values.

    Args:
        results: The dictionary to update.
        target_keys: A tuple representing the nested path to the value.
        value: The new value to set.
    """
    if value is None: return
    d = results
    for key in target_keys[:-1]: d = d.setdefault(key, {})
    final_key = target_keys[-1]
    if _get_priority(value) >= _get_priority(d.get(final_key)):
        d[final_key] = value

def _map_precision(value: Any) -> Optional[str]:
    """
    Normalizes model precision strings to a standard uppercase format.
    
    For example, 'torch.bfloat16' or 'bfloat16' would be converted to 'BF16'.

    Args:
        value: The original precision string or a regex match object.

    Returns:
        The normalized precision string, or None if conversion is not possible.
    """
    precision_map = {'bfloat16': 'BF16', 'float16': 'FP16', 'fp8': 'FP8'}
    if value is None: return None
    val_str = value.group(1) if hasattr(value, 'group') else str(value)
    key = val_str.split('.')[-1]
    return precision_map.get(key, key.upper())

def _to_int_safe(value: Any) -> Optional[int]:
    """
    Safely converts a value to an integer.

    Handles strings, numbers, or regex match objects. Returns None on failure 
    instead of raising an exception.

    Args:
        value: The value to convert.

    Returns:
        The converted integer, or None on failure.
    """
    if value is None: return None
    try:
        val_str = value.group(1) if hasattr(value, 'group') else str(value)
        return int(val_str)
    except (ValueError, TypeError, IndexError):
        return None

def _to_str_safe(value: Any) -> Optional[str]:
    """
    Safely converts a value to a string.

    Specifically handles regex match objects by attempting to extract their first
    captured group.

    Args:
        value: The value to convert.

    Returns:
        The converted string, or None on failure.
    """
    if value is None: return None
    if hasattr(value, 'group'):
        try:
            return value.group(1)
        except IndexError:
            return str(value)
    return str(value)

def _extract_arch_from_metadata(value: Any) -> Optional[str]:
    """
    Extracts the GPU compute architecture (e.g., 'sm80') from metadata.
    
    Supports various formats like 'arch=\'sm90\'' or 'device_capability=8.0'.

    Args:
        value: A string, dict, or regex match object containing architecture info.

    Returns:
        The extracted architecture string, or None if not found.
    """
    if value is None: return None
    val_str = value.group(1) if hasattr(value, 'group') and value.groups() else str(value)
    match = re.search(r"arch='(sm\d+)'", val_str)
    if match: return match.group(1)
    cap_match = re.search(r"device_capability=(\d+\.\d+)", val_str)
    if cap_match:
        capability = float(cap_match.group(1))
        cap_map = {9.0: "Hopper (sm90)", 8.0: "Ampere (sm80)", 8.6: "Ampere (sm86)"}
        return cap_map.get(capability, f"Capability {capability}")
    if isinstance(value, dict) and (arch := value.get('arch')): return arch
    return None

def _extract_fp8_from_bool(value: Any) -> Optional[str]:
    """
    Returns 'FP8' if the value is True.

    This is used to handle boolean flags that indicate whether FP8 is enabled.

    Args:
        value: A boolean value or its string representation.

    Returns:
        'FP8' if the value is True, otherwise None.
    """
    if value is True or (isinstance(value, str) and value.lower() == 'true'):
        return 'FP8'
    return None

def _search_dict_fuzzy(data: Any, key_pattern: re.Pattern) -> Optional[Any]:
    """
    Recursively searches a nested dict or list for a key matching a regex pattern.

    This function performs a deep traversal and returns the value of the first key
    that fully matches the `key_pattern`.

    Args:
        data: The dictionary or list to search.
        key_pattern: The compiled regex object to match against keys.

    Returns:
        The value of the first matching key found, or None if no match is found.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key_pattern.fullmatch(key): return value
            found = _search_dict_fuzzy(value, key_pattern)
            if found is not None: return found
    elif isinstance(data, list):
        for item in data:
            found = _search_dict_fuzzy(item, key_pattern)
            if found is not None: return found
    return None

# --- SGLang Parser ---
def _extract_for_sglang(log_content: str) -> dict:
    """
    Extracts configuration information from SGLang framework logs.

    This parser is designed for SGLang's JSON-per-line log format. It iterates
    through the log to extract model details, server arguments, parallel config,
    and request information from specific JSON structures.

    Args:
        log_content: The full string content of the SGLang log.

    Returns:
        A dictionary containing the extracted information.
    """
    results = {
        'Framework Name': 'sglang',
        'Model Name': None,
        'Model Precision': None,
        'Model Layers': None,
        'Startup Config': None,
        'batch_sizes': [],
        'input_lengths': [],
        'parallel_config': {}
    }
    
    request_lengths = {}
    req_info_re = re.compile(r"Req\(rid=([a-f0-9]{32}),\s*input_ids=\[([\d,\s]*)\]")

    for i, line in enumerate(StringIO(log_content)):
        # Extract request input lengths from non-standard JSON lines.
        if '"attr":"reqs"' in line:
            matches = req_info_re.findall(line)
            for rid, ids_str in matches:
                if rid not in request_lengths:
                    try:
                        length = len([num for num in ids_str.split(',') if num.strip()])
                        request_lengths[rid] = length
                    except:
                        pass
            continue

        try:
            data = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue

        kunlun_attr = data.get('kunlunClassAttr', {})
        if not isinstance(kunlun_attr, dict): continue

        arg_name = kunlun_attr.get('arg_name')
        attr_name = kunlun_attr.get('attr')

        if arg_name == 'server_args':
            server_args = data.get('ClassObjectJson')
            if isinstance(server_args, dict):
                if model_path := server_args.get('model_path'):
                    results['Model Name'] = str(model_path).strip('/').split('/')[-1]
                if tp_size := server_args.get('tp_size'):
                    results['parallel_config']['tp_size'] = tp_size
                if dtype := server_args.get('dtype'):
                    if results.get('Model Precision') is None and dtype != 'auto':
                         results['Model Precision'] = str(dtype).upper()
                results['Startup Config'] = json.dumps(server_args)

        elif attr_name == 'model_config':
            model_config = data.get('ClassObjectJson')
            if isinstance(model_config, dict):
                if dtype := model_config.get('torch_dtype') or model_config.get('dtype'):
                    results['Model Precision'] = str(dtype).upper()
                if layers := model_config.get('num_hidden_layers'):
                    results['Model Layers'] = int(layers)
                if not results['Model Name'] and (name_path := model_config.get('_name_or_path')):
                     results['Model Name'] = str(name_path).strip('/').split('/')[-1]

        elif attr_name == 'batch_size':
            if 'Value' in data:
                try:
                    bs = int(data['Value'])
                    results['batch_sizes'].append(bs)
                except (ValueError, TypeError):
                    pass
                            
    if request_lengths:
        results['input_lengths'] = list(request_lengths.values())

    return results

def _extract_for_vllm(log_content: str) -> dict:
    """
    Extracts configuration information from vLLM framework logs.

    This parser is designed for vLLM's JSON-per-line log format and supports
    both modern and legacy versions to extract batch sizes and input lengths.

    Supported Formats:
    1. Modern (v1) Logs:
       - Extracts batch info from 'GPUModelRunner' logs with 'requests' attribute.
       - Batch size is the number of requests in the `ClassObjectJson` dictionary.
       - Input length is parsed from `prompt_token_ids` in each request's state.
    2. Legacy Logs:
       - Extracts batch size from 'SchedulerOutputs' with 'scheduled_seq_groups'.
       - Infers input length by correlating 'request_id' and 'prompt_token_ids'
         from 'SelfAttnBlockSpaceManager' log entries.

    Args:
        log_content: The full string content of the vLLM log.

    Returns:
        A dictionary containing the extracted information.
    """
    results = {
        'Framework Name': 'vllm', 'Model Name': None, 'Model Precision': None, 'Model Layers': None,
        'Startup Config': "Not available from provided vllm log format",
        'batch_sizes': [], 'input_lengths': [], 'parallel_config': {},
        'Hardware Info': {}, 'Cache Sizes': {}
    }
    
    detected_frameworks = {'vllm'}
    
    # State and regex for parsing
    request_lengths = {}
    last_req_id_set = None
    batch_req_id_pattern = re.compile(r"request_id=([\w\d\.\-]+)")
    pending_id = None
    pending_ids = None
    input_req_id_pattern = re.compile(r'"Value":\s*"?([\w\d\.\-]+)"?\s*\}')
    scheduled_groups_pattern = re.compile(r'"Value":\s*(\[.*\])\s*\}')
    token_ids_pattern = re.compile(r'"Value":\s*(\[[\d,\s]*\])\s*\}')
    prompt_len_re = re.compile(r"prompt_token_ids=\[([\d,\s]*)\]")


    for line_num, line in enumerate(StringIO(log_content), 1):
        
        try:
            data = json.loads(line)
            kunlun_attr = data.get('kunlunClassAttr', {})
            if not isinstance(kunlun_attr, dict):
                continue
            
            module = kunlun_attr.get('module', '')
            if "vllm" not in module:
                continue
            
            class_name = kunlun_attr.get('class')
            attr_name = kunlun_attr.get('attr')
            value = data.get('Value')

            # Modern v1 log format parsing
            if class_name == 'GPUModelRunner' and attr_name == 'requests':
                class_obj_json = data.get('ClassObjectJson')
                if isinstance(class_obj_json, dict):
                    batch_size = len(class_obj_json)
                    if batch_size > 0:
                        results['batch_sizes'].append(batch_size)
                    
                    for req_id, state_str in class_obj_json.items():
                        if req_id not in request_lengths and isinstance(state_str, str):
                            match = prompt_len_re.search(state_str)
                            if match:
                                try:
                                    length = len([tok for tok in match.group(1).split(',') if tok.strip()])
                                    request_lengths[req_id] = length
                                except:
                                    pass
                continue

            # Legacy log format parsing for backward compatibility
            elif class_name == 'SchedulerOutputs' and attr_name == 'scheduled_seq_groups':
                value_str = str(value) if value is not None else ""
                req_ids = batch_req_id_pattern.findall(value_str)
                current_req_id_set = set(req_ids)
                if current_req_id_set and current_req_id_set != last_req_id_set:
                    batch_size = len(current_req_id_set)
                    if batch_size > 0:
                        results['batch_sizes'].append(batch_size)
                    last_req_id_set = current_req_id_set

            elif class_name == 'SelfAttnBlockSpaceManager' and attr_name == 'request_id':
                if value is not None:
                    current_id = str(value).strip()
                    if pending_ids is not None:
                        if current_id not in request_lengths:
                            request_lengths[current_id] = len(pending_ids)
                        pending_ids = None
                    else:
                        pending_id = current_id

            elif class_name == 'SelfAttnBlockSpaceManager' and attr_name == 'prompt_token_ids':
                if isinstance(value, list):
                    if pending_id is not None:
                        if pending_id not in request_lengths:
                            request_lengths[pending_id] = len(value)
                        pending_id = None
                    else:
                        pending_ids = value

            elif class_name == 'VllmConfig' and attr_name == '__str__':
                if isinstance(value, str):
                    results['Startup Config'] = value

            elif attr_name == 'model_config':
                model_config = data.get('ClassObjectJson')
                if isinstance(model_config, dict):
                    if model_path := model_config.get('_name_or_path'):
                        results['Model Name'] = str(model_path).strip('/').split('/')[-1]
                    if dtype := model_config.get('torch_dtype'):
                        results['Model Precision'] = str(dtype).upper()
                    if layers := model_config.get('num_hidden_layers'):
                        results['Model Layers'] = int(layers)
                    if max_len := model_config.get('max_position_embeddings'):
                        results['Cache Sizes']['Input Length (max_model_length)'] = int(max_len)
                    if model_config.get("transformers_version"):
                        detected_frameworks.add("transformers")

        except json.JSONDecodeError:
            # Fallback path for non-standard JSON in legacy logs
            if '"attr":"scheduled_seq_groups"' in line:
                match = scheduled_groups_pattern.search(line)
                if match:
                    req_ids = batch_req_id_pattern.findall(match.group(1))
                    current_req_id_set = set(req_ids)
                    if current_req_id_set and current_req_id_set != last_req_id_set:
                        bs = len(current_req_id_set)
                        if bs > 0: results['batch_sizes'].append(bs)
                        last_req_id_set = current_req_id_set

            elif '"attr":"request_id"' in line and '"class":"SelfAttnBlockSpaceManager"' in line:
                match = input_req_id_pattern.search(line)
                if match:
                    current_id = match.group(1).strip()
                    if pending_ids is not None:
                        if current_id not in request_lengths: request_lengths[current_id] = len(pending_ids)
                        pending_ids = None
                    else:
                        pending_id = current_id

            elif '"attr":"prompt_token_ids"' in line and '"class":"SelfAttnBlockSpaceManager"' in line:
                match = token_ids_pattern.search(line)
                if match:
                    try:
                        token_ids = json.loads(match.group(1))
                        if isinstance(token_ids, list):
                            if pending_id is not None:
                                if pending_id not in request_lengths: request_lengths[pending_id] = len(token_ids)
                                pending_id = None
                            else:
                                pending_ids = token_ids
                    except json.JSONDecodeError:
                        pass

            elif '"class":"VllmConfig"' in line and '"attr":"__str__"' in line:
                 config_match = re.search(r'"Value":\s*(.*)\s*\}\s*$', line)
                 if config_match:
                     results['Startup Config'] = config_match.group(1).strip()
    
    if request_lengths:
        results['input_lengths'] = sorted(list(set(request_lengths.values())))

    results['Framework Name'] = " / ".join(sorted(list(detected_frameworks)))
    return results

# --- Legacy/Generic Parser ---
def _extract_for_legacy(log_content: str) -> dict:
    """
    Extracts configuration info from generic or legacy framework logs.

    This parser uses a general-purpose strategy that does not rely on a fixed log
    structure. It scans each line of the log against a predefined set of rules
    (`LEGACY_EXTRACTION_RULES`). Each rule combines a JSON key pattern (for JSON
    lines) and a regex pattern (for plain text lines) to flexibly find information
    across different log formats.

    Args:
        log_content: The full string content of the log.

    Returns:
        A dictionary containing the extracted information.
    """
    # Each rule defines: a target path in the results dict, a fuzzy JSON key 
    # pattern, a plain text regex pattern, and a processor function to clean the value.
    LEGACY_EXTRACTION_RULES = [
        (('Model Name',), re.compile(r'model_path|name_or_path|served_model_name'), re.compile(r"['\"](model_path|name_or_path|served_model_name)['\"]\s*[:=]\s*['\"]([^'\",}]+)['\"]"), _to_str_safe),
        (('Model Layers',), re.compile(r'num_hidden_layers|n_layers|num_layers'), re.compile(r"['\"](num_hidden_layers|n_layers|num_layers)['\"]\s*[:=]\s*(\d+)"), _to_int_safe),
        (('Hardware Info', 'Chip Model'), re.compile(r'device_name'), re.compile(r"device_name='([^']+)'"), _to_str_safe),
        (('Hardware Info', '_arch'), re.compile(r'metadata|config'), re.compile(r"(arch='sm\d+'|device_capability=[\d\.]+)"), _extract_arch_from_metadata),
        (('Startup Config',), re.compile(r'server_args'), re.compile(r"['\"]server_args['\"]\s*[:=]\s*(\"ServerArgs\(.+?\)\")"), _to_str_safe),
        (('Cache Sizes', 'Input Length (max_model_length)'), re.compile(r'model_max_length|max_position_embeddings'), re.compile(r"['\"](model_max_length|max_position_embeddings)['\"]\s*[:=]\s*(\d+)"), _to_int_safe),
        (('Cache Sizes', 'KVCache Length (max_total_tokens)'), re.compile(r'max_total_num_tokens|max_total_tokens'), re.compile(r"['\"](max_total_num_tokens|max_total_tokens)['\"]\s*[:=]\s*(\d+)"), _to_int_safe),
        (('Cache Sizes', 'Max Prefill Tokens'), re.compile(r'max_prefill_tokens'), re.compile(r"['\"]max_prefill_tokens['\"]\s*[:=]\s*(\d+)"), _to_int_safe),
        (('precision_details', 'Params'), re.compile(r'dtype|torch_dtype|param_dtype'), re.compile(r"['\"](dtype|torch_dtype|param_dtype)['\"]\s*[:=]\s*['\"]?([\w\.]+)['\"]?"), _map_precision),
        (('precision_details', 'GEMM'), re.compile(r'fp8_gemm'), re.compile(r"fp8_gemm=([Tt]rue)"), _extract_fp8_from_bool),
    ]
    # Rules specifically for extracting parallel configuration parameters.
    PARALLEL_RULES = {
        key: (target_keys, re.compile(f'["\']{key}["\']\\s*[:=]\\s*(\\d+)'))
        for key, target_keys in [
            ('tp_size', ('parallel_config', 'tp_size')), ('pp_size', ('parallel_config', 'pp_size')),
            ('dp_size', ('parallel_config', 'dp_size')), ('nnodes', ('parallel_config', 'nnodes')),
            ('device_count', ('Hardware Info', 'Chip Count')), ('world_size', ('Hardware Info', 'Chip Count')),
        ]
    }
    
    results = {
        'Framework Name': None, 'Model Name': None, 'Model Precision': None, 'Model Layers': None,
        'Hardware Info': {}, 'Startup Config': None, 'Cache Sizes': {},
        'Request Param Analysis': set(), 'requests_raw': [], 'parallel_config': {},
        'precision_details': {},
    }
    detected_frameworks = set()

    for line in StringIO(log_content):
        data = None
        try: data = json.loads(line)
        except (json.JSONDecodeError, TypeError): pass

        # Identify Python modules/frameworks from JSON log entries.
        if data:
            if module_path := _search_dict_fuzzy(data, re.compile(r"module")):
                if isinstance(module_path, str) and module_path != "(null)":
                    detected_frameworks.add(module_path.split('.')[0])
        
        # Apply generic extraction rules.
        for target_keys, json_pattern, str_pattern, processor in LEGACY_EXTRACTION_RULES:
            value = _search_dict_fuzzy(data, json_pattern) if data and json_pattern else None
            if _get_priority(value) < 4 and str_pattern:
                if match := str_pattern.search(line): value = match
            if value is not None:
                _update_results_legacy(results, target_keys, processor(value))
        
        # Apply parallel config extraction rules.
        for key, (target_keys, str_pattern) in PARALLEL_RULES.items():
            value = _search_dict_fuzzy(data, re.compile(f"^{key}$")) if data else None
            if value is None and (match := str_pattern.search(line)): value = match.group(1)
            if value is not None: _update_results_legacy(results, target_keys, _to_int_safe(value))

    # --- Post-processing and data cleanup ---

    # Determine framework name based on detected modules.
    core_frameworks = {'torch', 'tensorflow', 'sglang', 'transformers', 'vllm'}
    core_found = sorted([fw.capitalize() for fw in detected_frameworks if fw in core_frameworks])
    other_found = sorted([fw.capitalize() for fw in detected_frameworks if fw not in core_frameworks])
    if core_found or other_found: results['Framework Name'] = " / ".join(core_found + other_found)
    
    # Calculate total chip count from parallel config if not specified directly.
    hw_info = results.get('Hardware Info', {})
    if hw_info.get('Chip Count') is None:
        p_config = results.get('parallel_config', {})
        total_chips = p_config.get('tp_size', 1) * p_config.get('pp_size', 1) * p_config.get('dp_size', 1) * p_config.get('nnodes', 1)
        if total_chips > 1: hw_info['Chip Count'] = total_chips
        
    # Consolidate various precision details into a single descriptive string.
    if prec_details := results.get('precision_details', {}):
        if parts := [f"{v} ({k})" for k, v in prec_details.items() if v]:
            results['Model Precision'] = " / ".join(sorted(parts))

    return results

# --- Main Entry Point ---
def extract_model_info_from_log(file_path: str) -> Optional[dict]:
    """
    Extracts model and environment configuration from a log file.

    This function serves as the main entry point for the module. It reads a log
    file and automatically selects the appropriate parser (SGLang, vLLM, or
    generic) based on the log content's characteristics.

    Args:
        file_path: The path to the log file.

    Returns:
        A dictionary containing all extracted information, or None if the file
        cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except (FileNotFoundError, IOError) as e:
        return None

    # Prioritize specific framework parsers before falling back to the generic one.
    if "vllm.core.scheduler" in log_content or "vllm.worker.model_runner" in log_content or "vllm.v1.worker.gpu_model_runner" in log_content:
        return _extract_for_vllm(log_content)
    
    if "sglang.srt" in log_content:
        return _extract_for_sglang(log_content)
    
    return _extract_for_legacy(log_content)

def print_results(extracted_info: dict):
    """
    Prints the extracted information to the console in a user-friendly format.

    Args:
        extracted_info: A dictionary returned by `extract_model_info_from_log`.
    """
    if not extracted_info:
        print("Could not extract any valid information.")
        return
    
    print("Information extracted from the file:\n")
    print(f"* Framework Name: {extracted_info.get('Framework Name', 'Unknown')}")
    print(f"* Model Name: {extracted_info.get('Model Name', 'Not Found')}")
    print(f"* Model Precision: {extracted_info.get('Model Precision', 'Not Found')}")
    print(f"* Model Layers: {extracted_info.get('Model Layers', 'Not Found')}")
    
    hw_info = extracted_info.get('Hardware Info', {})
    if hw_info:
        print("* Hardware Info:")
        print(f"    * Chip Model: {hw_info.get('Chip Model', 'Not Found')}")
        print(f"    * Chip Count: {hw_info.get('Chip Count', 'Not Found')}")
    
    if extracted_info.get('Startup Config'): 
        print(f"* Startup Config: {extracted_info.get('Startup Config')}")

    cache_info = extracted_info.get('Cache Sizes', {})
    if cache_info:
        print("* Cache Sizes:")
        for key, value in cache_info.items(): 
            print(f"    * {key}: {value}")
    
    if extracted_info.get('batch_sizes'):
        print(f"* Batch Sizes: {sorted(list(set(extracted_info['batch_sizes'])))}")
    
    if extracted_info.get('input_lengths'):
        print(f"* Input Lengths: {sorted(list(set(extracted_info['input_lengths'])))}")


if __name__ == '__main__':
    """
    This block executes when the script is run directly.
    
    It is intended for development and testing, verifying the extraction logic
    by analyzing example log files for different frameworks.
    
    To use, replace the file paths below with your actual log files and run
    this Python script.
    """

    # Test SGLang log
    print("--- Analyzing SGLang log (model_config_sglang) ---")
    sglang_info = extract_model_info_from_log('D:\\analysis_files\\test-0-model_config.txt')
    print_results(sglang_info)
    
    print("\n" + "="*50 + "\n")

    # Test vLLM log (example 1)
    print("--- Analyzing vLLM 0.7.2 log (model_config_vllm) ---")
    vllm_info = extract_model_info_from_log('D:\\analysis_vllm\\test-0-model_config.txt')
    print_results(vllm_info)
    
    print("\n" + "="*50 + "\n")

    # Test vLLM log (example 2)
    print("--- Analyzing vLLM 0.10.0 log (model_config_vllm) ---")
    vllm_info = extract_model_info_from_log('D:\\problem\\test-0-model_config.txt')
    print_results(vllm_info)
    
    print("\n" + "="*50 + "\n")
