"""Descriptive command metadata only; never imports handlers or validates commands.

Maintain alongside src/commands parsers. Choices enumerate finite keywords;
empty choices denotes an open-ended argument, not an unavailable argument.
"""
from copy import deepcopy


def choice(value, *aliases):
    return {'value': value, 'aliases': list(aliases)}


def argument(name, description, *choices):
    return {'name': name, 'description': description, 'choices': list(choices)}


def entry(summary, *arguments):
    return {'summary': summary, 'arguments': list(arguments)}


EXPRESSION = ('Boolean node selection: quoted headers, #LABEL#, @file@, $sele$, '
              'residue positions, or {property comparison}; combine with &, |, !, ^. '
              'No spaces inside selection expressions. Referenced data must exist; '
              'negative residue positions use parentheses, e.g. K(-1).')
CLUSTER_MODE = argument('mode', 'Default leiden. Leiden requires graspologic-native; MCL requires markov_clustering, networkx and scipy.',
                        choice('leiden'), choice('mcl'), choice('jaccard'))
CLUSTER_PARAMETER = argument('parameter', 'Optional number: Leiden resolution defaults to 1.0; MCL inflation to 2.0; Jaccard threshold to 0.2.')
MIN_SIZE = argument('min_size', 'Optional integer, default 10. Smaller subsets become noise.')

COMMAND_METADATA = {
    'agent': entry('Open or configure the Agent interface; MCP blocks nested model requests.',
        argument('action', 'Omit to open the interface; off deactivates the model. Registration only does not open it.', choice('off', 'deactivate'), choice('--register-only')),
        argument('model_or_message', 'An existing model card name enclosed in <...> activates that model. Natural-language messages are supported manually but blocked through MCP.')),
    'alignment': entry('Load another alignment and report coverage; restore the previous alignment on loader failure.',
        argument('filename', 'Optional FASTA/HDF5 MSA filename or path; omission opens a file dialog requiring a visible Viewer. Partial and zero node overlap are accepted.')),
    'cluster': entry('Identify topology clusters and number retained clusters by size.',
        argument('action', 'Use list for current cluster statistics; otherwise perform clustering.', choice('list')),
        CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE),
    'color': entry('Apply color, scale, or shape assignments to visible nodes.',
        argument('action', 'reset restores configured node colors.', choice('reset')),
        argument('expression', EXPRESSION + ' Omission targets selected nodes; assignments may be chained.'),
        argument('color', 'Optional Matplotlib color name or hex color; at least one visual attribute is needed.'),
        argument('scale', 'Optional x-suffixed numeric multiplier, e.g. 2x or 0.5x.'),
        argument('shape', 'Optional VisPy marker token; circle and triangle are explicit parser aliases.',
            choice('disc', 'circle'), choice('arrow'), choice('ring'), choice('clobber'), choice('square'), choice('x'),
            choice('diamond'), choice('vbar'), choice('hbar'), choice('cross'), choice('tailed_arrow'),
            choice('triangle_up', 'triangle'), choice('triangle_down'), choice('star'), choice('cross_lines'),
            *[choice(s) for s in ('o', '+', '++', 's', '-', '|', '->', '>', '^', 'v', '*')])),
    'esmfold': entry('Open the structure viewer or submit selected sequences for ESM3 folding.',
        argument('options', 'Omit with no selection to open the structure viewer. Folding defaults to local ESM3 and exactly one selected node; multi permits multiple nodes, large uses Biohub. large and multi may be combined in either order; duplicates are invalid.',
            choice('large'), choice('multi'), choice('--register-only'))),
    'export': entry('Export in-memory sequences as FASTA subsets beneath Analysis Results.',
        argument('target', 'Default clusters, excluding noise; groups exports all custom groups. Requires corresponding memberships and in-memory sequences. Do not mix these modes with explicit labels.', choice('clusters'), choice('groups', 'group')),
        argument('labels', 'One or more #LABEL# tokens for existing clusters, groups, or noise; repeated labels are deduplicated. Legacy group: prefixes are rejected.')),
    'group': entry('Assign overlapping custom group labels or manage existing groups.',
        argument('action', 'Omit to assign names; list prints statistics, remove deletes named groups, reset clears all groups.', choice('list'), choice('remove', 'delete'), choice('reset')),
        argument('expression', EXPRESSION + ' Pair each expression with a group name; a single name targets selected nodes.'),
        argument('names', 'Names use letters, digits, underscores, hyphens or periods, without spaces. Reserved command names and existing canonical cluster/subcluster labels cannot be assigned. remove accepts multiple names.')),
    'hide': entry('Hide selected or matching visible nodes and their connected edges.',
        argument('action', 'single hides nodes with no active edges at the current threshold; reset unhides all nodes. Without arguments, hide selected nodes.', choice('single', 'free'), choice('reset')),
        argument('expression', EXPRESSION)),
    'label': entry('Queue an XLSX subset-conservation report; requires a nonempty MSA, active reference and background scheduler.',
        argument('action', 'reset clears cluster labels without running analysis.', choice('reset')),
        argument('target', 'Omit to analyze all available cluster/group results; an explicit target requires its memberships.', choice('clusters', 'cluster'), choice('groups', 'group')),
        argument('keys', 'Each key takes a number. gmax defaults to 40%, cmin to 98%, identity reweighting is off by default. gmax/cmin accept fractions or percentages; id also accepts percentage numbers. gmin is fixed at 97% and cannot be supplied.', choice('gmax', 'global_max', 'g_max'), choice('cmin', 'cluster_min', 'c_min'), choice('id')),
        argument('numbers', 'Alternatively supply gmax, cmin, then optional identity positionally. Do not put positional numbers after keyword arguments.'),
        argument('filename', 'Optional final XLSX basename; extension added. Numeric or reserved names must include .xlsx. Explicit filenames overwrite; automatic timestamp names avoid collisions.')),
    'logo': entry('Queue a sequence-logo image for a selected subset; requires a nonempty MSA and background scheduler.',
        argument('positions', 'Required bracketed reference positions/ranges, e.g. [1,5-8,10.1] or [(-3)-(-1)]. Fractional insertions must be explicit.'),
        argument('expression', EXPRESSION + ' Defaults to selected nodes, or all nodes if nothing is selected.'),
        argument('filename', 'Optional SVG/PNG basename; default timestamped SVG. The last unrecognized string is interpreted as a filename.'),
        argument('mode', 'Default bits (information content); pcts displays frequencies.', choice('bits', 'bit'), choice('pcts', 'pct', 'percentage', 'percentages')),
        argument('gap_mode', 'Default with_gap scales height by occupancy.', choice('with_gap', 'with_gaps', 'gaps', 'gap'), choice('no_gap', 'no_gaps')),
        argument('preset', 'Optional standalone color preset, case-insensitive; default chemistry.', *[choice(s) for s in ('chemistry','classic','grays','base_pairing','colorblind_safe','weblogo_protein','skylign_protein','dmslogo_charge','dmslogo_funcgroup','hydrophobicity','charge','NajafabadiEtAl2017')]),
        argument('color_scheme', 'Alternatively color_scheme=NAME, colors=NAME, color=NAME or scheme=NAME passes a scheme to the renderer.'),
        argument('identity', 'Optional numeric redundancy threshold: 0.9, 90 or 90%; omitted means reweighting off.')),
    'meta': entry('Open the metadata spreadsheet, import/export metadata, or manage columns and the metadata HUD.',
        argument('action', 'Omit to open the spreadsheet. upload accepts files; download writes metadata; delete removes columns; show enables the HUD. A filename without upload also imports it.',
            choice('upload', 'import'), choice('download', 'retrieve', 'export'), choice('delete', 'remove', 'clear'), choice('show', 'display'), choice('--register-only')),
        argument('filenames', 'Upload one or more CSV/XLS/XLSX paths or names from the metadata directory. Download accepts an optional filename (CSV by default); automatic names avoid overwrite.'),
        argument('properties', 'delete requires one or more existing column names, case-insensitive; deleting all with all is unsupported. show requires one existing property.'),
        argument('display_action', 'Only after show/display: clear the metadata HUD.', choice('clear', 'off'))),
    'offset': entry('Inspect or change reference-position numbering without changing alignment columns.',
        argument('integer', 'Omit to inspect. A supplied integer offset requires a loaded alignment with active reference; default launch offset is configured separately.')),
    'print': entry('Save a rendered network snapshot as PNG or SVG.',
        argument('filename', 'Optional basename; defaults to timestamped PNG under Saved Images.'),
        argument('modifiers', 'transparent and full can combine for PNG; svg cannot combine with either. full stitches the network. PNG margins are trimmed with padding.', choice('transparent'), choice('full'), choice('svg'))),
    'query': entry('Print alignment-position distributions or search positions by residue-frequency logic; requires a nonempty MSA.',
        argument('expression', EXPRESSION + ' Defaults to selected nodes, otherwise all nodes.'),
        argument('positions_or_logic', 'Required brackets: positions/ranges (E or END denotes the last residue), or comparisons such as [(RHK)>50%]. Combined comparisons need outer parentheses, e.g. [((RHK)>50%)&((DE)>20%)]. GAP or _ denotes gaps. Frequencies divide by all mapped subset sequences, including gaps.')),
    'redo': entry('Reapply the last undone state; report a no-op if redo history is empty.'),
    'reference': entry('Inspect or change the reference sequence used for alignment mapping.',
        argument('target', 'Omit to inspect. Supply a partial header or wildcard pattern. A configured reference absent from the current MSA remains inactive in occupancy mode.')),
    'reset': entry('Restore selected network properties in one undoable action.',
        argument('targets', 'One or more targets: reset colors/sizes to configured defaults, shapes to discs, clear clusters/groups, unhide nodes, restore original/last-saved positions, or reset render order. Keywords are case-insensitive.',
            choice('colors','color'), choice('sizes','size'), choice('shapes','shape'), choice('clusters','cluster'), choice('groups','group'),
            choice('hide','hides','hidden','hiddens'), choice('network','networks'), choice('order','orders','layer','layers'))),
    'run': entry('Open a script-selection dialog and execute a text command batch or Python-generated commands.',
        argument('script', 'Chosen through a visible Viewer dialog, not a path argument. TXT lines become commands; Python stdout supplies commands. Nested run lines are ignored. MCP tracks child commands and Python workers.')),
    'save': entry('Save current network state as an HDF5 layout cache; requires a valid active cache manifest and provenance binding.',
        argument('filename', 'Optional .h5 basename in the active cache folder; omitted names use version_XX.h5. Stores positions, styling, visibility, render order, memberships and metadata.')),
    'select': entry('Modify the visible-node selection or save selected headers/sequences.',
        argument('mode', 'Default change. Modes may precede or follow the expression; invert takes no expression. save must come first and requires a filename.',
            choice('change'), choice('add','plus','include'), choice('subtract','minus','remove'), choice('filter','keep','intersect'), choice('invert'), choice('save')),
        argument('expression', EXPRESSION),
        argument('filename', 'Required for save: .txt saves headers, .fasta saves sequences; at least one selected node is needed.')),
    'spectrum': entry('Color visible nodes by a numerical metadata property.',
        argument('property', 'Required bare {PROPERTY_NAME} selector for a numerical metadata column; arguments may occur in any order.'),
        argument('expression', EXPRESSION + ' Defaults to all visible nodes.'),
        argument('color_scheme', 'Optional installed Matplotlib colormap name, default coolwarm; unknown names fall back to coolwarm with a warning.')),
    'subcluster': entry('Subcluster a topology cluster into custom group labels while retaining original cluster membership.',
        argument('action', 'clear removes subcluster group labels and leaves colors unchanged.', choice('clear')),
        argument('cluster_name', 'Required existing topology cluster name such as cluster_2 when not clearing.'),
        CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE),
    'undo': entry('Restore the previous state; report a no-op if undo history is empty.'),
    'zoom': entry('Set camera view width while retaining its center and canvas aspect ratio.',
        argument('width', 'Numeric width in scene units; omission prints help.')),
}

# Help flags are command-specific; these are documentation, not dispatch aliases.
for name, metadata in COMMAND_METADATA.items():
    help_description = 'Print command help instead of performing the operation.'
    if name == 'label':
        help_description += ' Requires a nonempty MSA and active reference, like analysis.'
    metadata['arguments'].append(argument('help', help_description,
        choice('help', '-h', '-?' if name in {'export', 'label'} else '--help')))


def get_command_metadata(name):
    """Return independent response data so callers cannot mutate the catalog."""
    return deepcopy(COMMAND_METADATA[name])
