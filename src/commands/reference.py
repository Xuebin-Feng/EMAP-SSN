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
import fnmatch

def run(viewer, args):
    if not args:
        current_ref = getattr(viewer, 'resolved_ref_full', None) or getattr(viewer, 'active_reference', 'None')
        msg = f"Current Reference: {current_ref}"
        Command_Engine.print_help(viewer, msg)
        return
        
    if args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: reference [TARGET]\nDescription: Changes the reference sequence for alignment mapping.\n  - Call without arguments to see the current active reference.\n  - Pass a partial sequence header name to set a new reference.\nExamples:\n  reference\n  reference SeqA"
        Command_Engine.print_help(viewer, msg)
        return

    target = args[0]
    target_lower = target.lower()
    found_ref = None
    found_ref_full = None

    matches = [h for h in viewer.full_headers if fnmatch.fnmatch(h.lower(), target_lower) or target_lower in h.lower()]
    if matches:
        found_ref = matches[0]
        found_ref_full = found_ref
        if len(matches) > 1:
            print(f"Warning: Multiple matches found for '{target}'. Using '{found_ref}'.")
    else:
        if getattr(viewer, 'alignment', None) and viewer.alignment.aln:
            for record in viewer.alignment.aln:
                k = record.id
                if fnmatch.fnmatch(k.lower(), target_lower) or target_lower in k.lower():
                    found_ref = k
                    found_ref_full = record.description if record.description else k
                    break
    
    if found_ref:
        viewer.active_reference = target
        if found_ref_full:
            viewer.resolved_ref_full = found_ref_full
        else:
            viewer.resolved_ref_full = found_ref 
            
        print(f"\nReloading alignment...")
        viewer.console_text.text = f"Reloading alignment with new reference: {target}..."
        
        viewer.load_global_alignment()
        
        if viewer.alignment and viewer.alignment.aln is not None:
            viewer.console_text.text = "Reference successfully set."
    else:
        err = f"Error: Reference '{target}' not found."
        viewer.console_text.text = err
        print(f"\n{err}")
