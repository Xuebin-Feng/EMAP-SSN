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
              'negative residue positions and negative metadata range bounds use '
              'parentheses, e.g. K(-1) or {GRAVY=(-1)-0}.')
CLUSTER_MODE = argument('mode', 'Default leiden. Leiden requires graspologic-native; MCL requires markov_clustering, networkx and scipy.',
                        choice('leiden'), choice('mcl'), choice('jaccard'))
CLUSTER_PARAMETER = argument('parameter', 'Optional number: Leiden resolution defaults to 1.0; MCL inflation to 2.0, within 1.1 to 10.0; Jaccard threshold to 0.2.')
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
        argument('color', 'Optional Matplotlib color name or hex color; at least one visual attribute is needed. A valid selection expression takes precedence, so C53 selects cysteine 53 and x2 selects residue X at position 2.'),
        argument('scale', 'Optional multiplier of the configured node size: a finite, non-negative number followed by x, e.g. 2x, 0.5x or 0x.'),
        argument('shape', 'Optional VisPy marker token; circle and triangle are explicit parser aliases.',
            choice('disc', 'circle'), choice('arrow'), choice('ring'), choice('clobber'), choice('square'), choice('x'),
            choice('diamond'), choice('vbar'), choice('hbar'), choice('cross'), choice('tailed_arrow'),
            choice('triangle_up', 'triangle'), choice('triangle_down'), choice('star'), choice('cross_lines'),
            *[choice(s) for s in ('o', '+', '++', 's', '-', '|', '->', '>', '^', 'v', '*')])),
    'esmfold': entry('Open the structure viewer or submit selected sequences for ESM3 folding.',
        argument('options', 'Omit with no selection to open the structure viewer. Folding defaults to local ESM3 and exactly one selected node; multi permits multiple nodes, large uses Biohub. large and multi may be combined in either order; duplicates are invalid.',
            choice('large'), choice('multi'), choice('--register-only'))),
    'export': entry('Export in-memory sequences as FASTA subsets beneath Analysis Results.',
        argument('target', 'Default clusters, excluding noise; groups exports all custom groups. Requires corresponding memberships and in-memory sequences. Do not mix these modes with explicit labels. Group labels name the files and clustering parameters the cluster folder; one that is not a plain filename (only a hand-edited layout cache can carry one) is refused before anything is written.', choice('clusters'), choice('groups', 'group')),
        argument('labels', 'One or more #LABEL# tokens for existing clusters, groups, or noise; repeated labels are deduplicated. Legacy group: prefixes are rejected.')),
    'group': entry('Assign overlapping custom group labels or manage existing groups.',
        argument('action', 'Omit to assign names; list prints statistics, remove deletes named groups (all of which must exist, or nothing is removed), reset clears all groups.', choice('list'), choice('remove', 'delete'), choice('reset')),
        argument('expression', EXPRESSION + ' Pair each expression with a group name; a single name targets selected nodes.'),
        argument('names', 'Names use letters, digits, underscores, hyphens or periods, without spaces. Reserved command names, canonical cluster_N names of clusters that currently exist, and every subcluster_N_M name in the generated form (positive IDs without leading zeros) cannot be assigned. remove accepts multiple names.')),
    'hide': entry('Hide selected or matching visible nodes and their connected edges.',
        argument('action', 'single hides nodes with no active edges at the current threshold; reset unhides all nodes. Without arguments, hide selected nodes.', choice('single', 'free'), choice('reset')),
        argument('expression', EXPRESSION)),
    'label': entry('Queue an XLSX subset-conservation report; requires a nonempty MSA, active reference and background scheduler.',
        argument('action', 'reset clears cluster labels without running analysis.', choice('reset')),
        argument('target', 'Omit to analyze all available cluster/group results; an explicit target requires its memberships.', choice('clusters', 'cluster'), choice('groups', 'group')),
        argument('keys', 'Each key takes a finite number written as a fraction (0.4) or a percentage (40 or 40%): a trailing % always means percent, and a bare value above 1 is a percentage. gmax defaults to 40%, cmin to 98%, identity reweighting is off by default. gmin is fixed at 97% and cannot be supplied.', choice('gmax', 'global_max', 'g_max'), choice('cmin', 'cluster_min', 'c_min'), choice('id')),
        argument('numbers', 'Alternatively supply gmax, cmin, then optional identity positionally. Do not put positional numbers after keyword arguments.'),
        argument('filename', 'Optional final XLSX basename; extension added. Numeric or reserved names must include .xlsx. Explicit filenames overwrite; automatic timestamp names avoid collisions.')),
    'logo': entry('Queue a sequence-logo image for a selected subset; requires a nonempty MSA and background scheduler.',
        argument('positions', 'Required bracketed displayed positions/ranges, e.g. [1,5-8,10.1] or [(-3)-(-1)]: reference numbering with an active reference, otherwise occupancy numbering as in query. Fractional insertions must be explicit. No spaces inside the brackets; ranges run from lower to higher.'),
        argument('expression', EXPRESSION + ' Defaults to selected nodes, or all nodes if nothing is selected.'),
        argument('filename', 'Optional SVG/PNG basename; default timestamped SVG. With two or more remaining strings the last is the filename; a single remaining string is a filename only if it ends in .svg or .png, otherwise it is the expression.'),
        argument('mode', 'Default bits (information content); pcts displays frequencies.', choice('bits', 'bit'), choice('pcts', 'pct', 'percentage', 'percentages')),
        argument('gap_mode', 'Default with_gap scales height by occupancy.', choice('with_gap', 'with_gaps', 'gaps', 'gap'), choice('no_gap', 'no_gaps')),
        argument('preset', 'Optional standalone color preset, case-insensitive; default chemistry.', *[choice(s) for s in ('chemistry','classic','grays','base_pairing','colorblind_safe','weblogo_protein','skylign_protein','dmslogo_charge','dmslogo_funcgroup','hydrophobicity','charge','NajafabadiEtAl2017')]),
        argument('color_scheme', 'Alternatively color_scheme=NAME, colors=NAME, color=NAME or scheme=NAME passes a scheme to the renderer.'),
        argument('identity', 'Optional numeric redundancy threshold: 0.9, 90 or 90%; omitted means reweighting off.')),
    'meta': entry('Open the metadata spreadsheet, import/export metadata, or manage columns and the metadata HUD.',
        argument('action', 'Omit to open the spreadsheet. upload accepts files; download writes metadata; delete removes columns; show enables the HUD. A filename without upload also imports it.',
            choice('upload', 'import'), choice('download', 'retrieve', 'export'), choice('delete', 'remove', 'clear'), choice('show', 'display'), choice('--register-only')),
        argument('filenames', 'Upload one or more CSV/XLS/XLSX paths or names from the metadata directory. Download accepts an optional plain filename, not a path, written to the metadata directory: .csv or .xlsx in any case, .csv when the extension is omitted; an explicit name overwrites, automatic names avoid overwrite.'),
        argument('properties', 'delete requires one or more existing column names, case-insensitive; deleting all with all is unsupported. show takes one property name; an unknown name prints a warning listing the available properties.'),
        argument('display_action', 'Only after show/display: clear the metadata HUD.', choice('clear', 'off'))),
    'offset': entry('Inspect or change reference-position numbering without changing alignment columns.',
        argument('integer', 'Omit to inspect. A supplied integer offset requires a loaded alignment with active reference; default launch offset is configured separately.')),
    'print': entry('Save a rendered network snapshot as PNG or SVG.',
        argument('filename', 'Optional plain filename under Saved Images, not a path; .png or .svg is added. Defaults to a timestamped PNG.'),
        argument('modifiers', 'transparent and full can combine for PNG; svg cannot combine with either. full stitches the network. PNG margins are trimmed with padding.', choice('transparent'), choice('full'), choice('svg'))),
    'query': entry('Print alignment-position distributions or search positions by residue-frequency logic; requires a nonempty MSA.',
        argument('expression', EXPRESSION + ' Defaults to selected nodes, otherwise all nodes.'),
        argument('positions_or_logic', 'Required brackets: positions/ranges (E or END denotes the last displayed position), or comparisons such as [(RHK)>50%]. Combined comparisons need outer parentheses, e.g. [((RHK)>50%)&((DE)>20%)]. GAP or _ denotes gaps. Frequencies divide by all mapped subset sequences, including gaps.')),
    'redo': entry('Reapply the last undone state; report a no-op if redo history is empty.'),
    'reference': entry('Inspect or change the reference sequence used for alignment mapping.',
        argument('target', 'Omit to inspect the current reference and whether it is active. Supply a full header, a leading identifier such as WP_0123.1, a partial header, or a wildcard pattern; an exact header or identifier takes priority, and the resolved full header anchors numbering. A configured reference absent from the current MSA remains inactive in occupancy mode.')),
    'reset': entry('Restore selected network properties in one undoable action.',
        argument('targets', 'One or more targets: reset colors/sizes to configured defaults, shapes to discs, clear clusters/groups, unhide nodes, restore original/last-saved positions, or reset render order. Keywords are case-insensitive; an unknown target fails the command without resetting anything.',
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
        argument('filename', 'Required for save: a plain filename in the header list directory, not a path. .txt saves headers, .fasta saves sequences, and other names get .txt; at least one selected node is needed.')),
    'spectrum': entry('Color visible nodes by a numerical metadata property.',
        argument('property', 'Required bare {PROPERTY_NAME} selector for a numerical metadata column; arguments may occur in any order.'),
        argument('expression', EXPRESSION + ' Defaults to all visible nodes.'),
        argument('color_scheme', 'Optional installed Matplotlib colormap name, default coolwarm; unknown names fall back to coolwarm with a warning. A token that is also a valid selection expression is read as the expression.')),
    'subcluster': entry('Subcluster a topology cluster into custom group labels, numbered by size, while retaining original cluster membership.',
        argument('action', 'clear removes the generated subcluster_N_M group labels, keeps custom groups with lookalike names such as subcluster_0_2, and leaves colors unchanged.', choice('clear')),
        argument('cluster_name', 'Required existing topology cluster name such as cluster_2 when not clearing.'),
        CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE),
    'undo': entry('Restore the previous state; report a no-op if undo history is empty.'),
    'zoom': entry('Set camera view width while retaining its center and canvas aspect ratio.',
        argument('width', 'Positive, finite width in scene units; omission prints help.')),
}

# Help flags are command-specific; these are documentation, not dispatch aliases.
for name, metadata in COMMAND_METADATA.items():
    help_description = 'Print command help instead of performing the operation.'
    if name == 'label':
        help_description += ' Requires a nonempty MSA and active reference, like analysis.'
    metadata['arguments'].append(argument('help', help_description,
        choice('help', '-h', '-?' if name in {'export', 'label'} else '--help')))


def policy(default, first_argument=None):
    """How a command in an Agent reply is handled, by its lower-case first
    argument ('' when there is none): 'run' at once, 'approve' only after the
    user presses Run in the Agent page, or 'refuse'."""
    return {'default': default, 'first_argument': dict(first_argument or {})}


# Commands that change only the view or the session's in-memory state run.
# Ones that write or import files, launch processes or dialogs, or spend
# Biohub credit wait for approval, and the agent may not reconfigure itself.
AGENT_POLICY = {
    'agent': policy('refuse'),
    'alignment': policy('approve'),
    'cluster': policy('run'),
    'color': policy('run'),
    'esmfold': policy('approve'),
    'export': policy('approve'),
    'group': policy('run'),
    'hide': policy('run'),
    'label': policy('approve', {'reset': 'run'}),
    'logo': policy('approve'),
    # A bare filename imports a file, like upload.
    'meta': policy('approve', dict.fromkeys(
        ('', 'show', 'display', 'delete', 'remove', 'clear', 'help', '-h', '--help', '--register-only'), 'run')),
    'offset': policy('run'),
    'print': policy('approve'),
    'query': policy('run'),
    'redo': policy('run'),
    'reference': policy('run'),
    'reset': policy('run'),
    'run': policy('approve'),
    'save': policy('approve'),
    'select': policy('run', {'save': 'approve'}),
    'spectrum': policy('run'),
    'subcluster': policy('run'),
    'undo': policy('run'),
    'zoom': policy('run'),
}
for name, metadata in COMMAND_METADATA.items():
    metadata['agent_policy'] = AGENT_POLICY[name]


def agent_policy(command):
    """'run', 'approve' or 'refuse' for one command line in an Agent reply.

    The command name is read the way Command_Engine dispatches it. A name the
    catalog does not know waits for approval, so a command added without a
    policy never runs unreviewed.
    """
    words = command.split()
    rule = AGENT_POLICY.get(words[0].lower() if words else '')
    if rule is None:
        return 'approve'
    return rule['first_argument'].get(words[1].lower() if len(words) > 1 else '', rule['default'])


def get_command_metadata(name):
    """Return independent response data so callers cannot mutate the catalog."""
    return deepcopy(COMMAND_METADATA[name])
