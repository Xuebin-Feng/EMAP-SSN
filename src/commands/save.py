# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import Command_Engine
import os
import h5py
import numpy as np
import json
import SSN_Config as cfg
import Cache_Manifest as cache_manifest
from utilities.Cache_Selection import resolve_selected_cache

def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: save [filename.h5]\nDescription: Takes a snapshot of the current network state (positions, colors, sizes, shapes, visibility, render order, clusters, groups) and saves it as an HDF5 layout cache.\nIf no filename is provided, it automatically generates a versioned filename (e.g., version_01.h5).\nExamples:\n  save\n  save my_layout.h5"
        Command_Engine.print_help(viewer, msg)
        return
        
    try:
        default_path, _ = resolve_selected_cache(cfg)
        folder_path = os.path.dirname(default_path)
        
        os.makedirs(folder_path, exist_ok=True)
        
        if args:
            save_name = args[0]
            if not save_name.endswith(".h5"):
                save_name += ".h5"
            cache_manifest.validate_cache_filename(save_name)
            final_save_path = os.path.join(folder_path, save_name)
        else:
            save_name = cache_manifest.next_cache_version_filename(folder_path)
            final_save_path = os.path.join(folder_path, save_name)

        manifest_id = getattr(viewer, 'cache_manifest_id', None)
        if not manifest_id:
            raise cache_manifest.CacheManifestError(
                "The active viewer has no cache manifest binding."
            )
        folder_manifest = cache_manifest.read_manifest(folder_path)
        if folder_manifest["manifest_id"] != manifest_id:
            raise cache_manifest.CacheManifestError(
                "The active cache folder manifest has changed."
            )
        partial_save_path = final_save_path + ".partial"
        if os.path.exists(partial_save_path):
            os.remove(partial_save_path)

        with h5py.File(partial_save_path, "w") as hf:
            hf.attrs["cache_manifest_id"] = manifest_id
            dt_str = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset("headers", data=np.array(viewer.full_headers, dtype=object), dtype=dt_str, compression="gzip")
            hf.create_dataset("positions", data=viewer.pos, compression="gzip")
            
            if hasattr(viewer, 'current_colors'): hf.create_dataset("colors", data=viewer.current_colors, compression="gzip")
            if hasattr(viewer, 'current_sizes'): 
                hf.create_dataset("sizes", data=viewer.current_sizes, compression="gzip")
                hf.attrs["base_node_size"] = cfg.NODE_SIZE
            if hasattr(viewer, 'current_shapes'): hf.create_dataset("shapes", data=np.array(viewer.current_shapes, dtype=object), dtype=dt_str, compression="gzip")
            if hasattr(viewer, 'visible_mask'): hf.create_dataset("visible_mask", data=viewer.visible_mask, compression="gzip")
            if hasattr(viewer, 'node_render_order'):
                render_order = cache_manifest.validate_node_render_order(
                    viewer.node_render_order, len(viewer.full_headers)
                )
                hf.create_dataset("node_render_order", data=render_order, compression="gzip")
            if getattr(viewer, 'cluster_labels', None) is not None: hf.create_dataset("cluster_labels", data=viewer.cluster_labels, compression="gzip")
            
            if hasattr(viewer, 'group_labels'):
                hf.create_dataset("group_labels", data=json.dumps([list(g) for g in viewer.group_labels]))
                
            if getattr(viewer, 'metadata', None):
                meta_group = hf.create_group("metadata")
                for prop_name, prop_data in viewer.metadata.items():
                    prop_type = prop_data["type"]
                    values = prop_data["values"]
                    
                    if prop_type == "number":
                        ds = meta_group.create_dataset(prop_name, data=values, compression="gzip")
                    else:
                        dt_str = h5py.string_dtype(encoding='utf-8')
                        ds = meta_group.create_dataset(prop_name, data=np.array(values, dtype=object), dtype=dt_str, compression="gzip")
                    ds.attrs["type"] = prop_type
                    
            # --- Save Custom Dynamic Attributes at Root Level ---
            if getattr(viewer, '_cacheable_attrs', None):
                for attr_name in viewer._cacheable_attrs:
                    if hasattr(viewer, attr_name):
                        val = getattr(viewer, attr_name)
                        if val is not None:
                            CORE_DATASETS = {
                                "headers", "positions", "colors", "sizes", "shapes", 
                                "visible_mask", "cluster_labels", "group_labels", "metadata",
                                "connectivity", "edge_scores", "node_render_order"
                            }
                            if attr_name in CORE_DATASETS:
                                continue
                                
                            if attr_name in hf:
                                del hf[attr_name]
                                
                            if isinstance(val, np.ndarray):
                                hf.create_dataset(attr_name, data=val, compression="gzip")
                            else:
                                ds = hf.create_dataset(attr_name, data=json.dumps(val))
                                ds.attrs["is_json"] = True
                
            if getattr(viewer, 'last_cluster_params', None) is not None: hf.attrs["last_cluster_params"] = json.dumps(viewer.last_cluster_params)

        os.replace(partial_save_path, final_save_path)
        
        if hasattr(viewer, 'original_pos'):
            viewer.original_pos = viewer.pos.copy()
        
        msg = f"State successfully saved: {save_name}"
        Command_Engine.print_help(viewer, msg)
        
    except Exception as e:
        if 'partial_save_path' in locals() and os.path.exists(partial_save_path):
            os.remove(partial_save_path)
        msg = f"Error saving layout state: {e}"
        Command_Engine.print_help(viewer, msg)
