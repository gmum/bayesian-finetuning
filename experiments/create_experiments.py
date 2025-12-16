import re, sys, os
import json
import itertools
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Generates scripts based on a template file and value ranges selected for placeholders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CONFIGURATION FILE FORMAT:
--------------------------
{
  "output_dir": "generated_files",
  "output_prefix": "experiment",
  "file_extension": ".sbatch",
  "replacements": {
    "FIXED_VALUE": "constant_value",
    "ANOTHER_FIXED": "another_constant"
  },
  "grid_search": {
    "LEARNING_RATE": [0.001, 0.01, 0.1],
    "BATCH_SIZE": [16, 32, 64]
  },
  "predefined_configs": [
    {
      "name": "baseline",
      "LEARNING_RATE": 0.001,
      "BATCH_SIZE": 32,
      "MODEL": "bert-base"
    }
  ]
}

TEMPLATE FORMAT:
----------------
Templates use $PLACEHOLDER_NAME format for replacements.
Special placeholder $IDENTIFIER (and $RAWIDENTIFIER) is automatically generated from all parameters.
Fields like $FILE,$DATE,$DATETIME,$DATE0,$DATETIME,etc. can be used inside of $IDENTIFIER/$RAWIDENTIFIER,
for example, '$FILE_$DATE0'

Example template:
  python train.py --lr $LEARNING_RATE --batch $BATCH_SIZE --run_name $RAWIDENTIFIER

NOTES:
------
- You can use 'grid_search' OR 'predefined_configs' OR BOTH
- All parameter values are converted to strings during replacement
- $IDENTIFIER is auto-generated as: prefix_PARAM1_VALUE1_PARAM2_VALUE2_...
""",
    )

    parser.add_argument("template", help="Path to the template file")

    parser.add_argument("config", help="Path to the configuration JSON file")

    parser.add_argument(
        "-i",
        "--id_prefix",
        default=None,
        help="Prefix for $IDENTIFIER placeholder (default: empty string)",
    )

    parser.add_argument(
        "-p",
        "--file_prefix",
        default=None,
        help="Prefix for names of generated files.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Prefix for output directory name.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with additional details",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )

    parser.add_argument(
        "-d",
        "--delimiter",
        default=None,
        type=str,
        required=False,
        help='Delimiter for string values (default: double quote "). Use empty string e.g. '
        " for no delimiter.",
    )

    return parser


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"{title}")
    print(f"{char * 70}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 70)


def load_config(
    config_path: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load and validate configuration from JSON file."""
    print_subsection("Loading Configuration")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        print(f"✓ Successfully loaded config from: {config_path}")

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a JSON object")

        # Set defaults
        config.setdefault("file_extension", None)
        config.setdefault("replacements", {})
        config.setdefault("grid_search", {})
        config.setdefault("predefined_configs", [])

        # Validate replacements
        if not isinstance(config["replacements"], dict):
            raise ValueError("'replacements' must be a dictionary")

        # Validate grid_search
        if not isinstance(config["grid_search"], dict):
            raise ValueError("'grid_search' must be a dictionary")

        for key, values in config["grid_search"].items():
            if not isinstance(values, list):
                raise ValueError(f"Grid search parameter '{key}' must be a list")
            if len(values) == 0:
                raise ValueError(f"Grid search parameter '{key}' cannot be empty")

        # Validate predefined_configs
        if not isinstance(config["predefined_configs"], list):
            raise ValueError("'predefined_configs' must be a list")

        for idx, preset in enumerate(config["predefined_configs"]):
            if not isinstance(preset, dict):
                raise ValueError(
                    f"Predefined config at index {idx} must be a dictionary"
                )

        # Check if at least one generation method is specified
        has_grid = bool(config["grid_search"])
        has_predefined = bool(config["predefined_configs"])

        if not has_grid and not has_predefined:
            raise ValueError(
                "Config must specify either 'grid_search' or 'predefined_configs' (or both)"
            )

        print(f"✓ Configuration validated successfully")

        if verbose:
            print(f"  Config structure:")
            print(f"    - replacements: {len(config['replacements'])} entries")
            print(f"    - grid_search: {len(config['grid_search'])} parameters")
            print(
                f"    - predefined_configs: {len(config['predefined_configs'])} configs"
            )

        return config

    except FileNotFoundError:
        print(f"✗ Error: Config file '{config_path}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON in config file:")
        print(f"  Line {e.lineno}, Column {e.colno}: {e.msg}")
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error loading config: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_template(template_path: str, verbose: bool = False) -> str:
    """Load template file and validate."""
    print_subsection("Loading Template")

    try:
        with open(template_path, "r") as f:
            template = f.read()

        print(f"✓ Successfully loaded template from: {template_path}")
        print(
            f"  Template size: {len(template)} characters, {len(template.splitlines())} lines"
        )

        # Detect placeholders in template (format: $PLACEHOLDER_NAME)
        placeholders = re.findall(r"\$([A-Z_][A-Z0-9_]*)", template)
        unique_placeholders = set(placeholders)

        if unique_placeholders:
            print(f"  Detected {len(unique_placeholders)} unique placeholder(s):")
            for ph in sorted(unique_placeholders):
                count = placeholders.count(ph)
                print(f"    - ${ph} (used {count} time(s))")
        else:
            print(f"  ⚠ Warning: No placeholders detected in template")

        if verbose and template.strip():
            print(f"  First 200 characters of template:")
            print(f"    {template[:200]}...")

        return template

    except FileNotFoundError:
        print(f"✗ Error: Template file '{template_path}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading template file: {e}")
        traceback.print_exc()
        sys.exit(1)


def generate_grid_combinations(grid_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations from grid search parameters."""
    if not grid_params:
        return []

    keys = list(grid_params.keys())
    values = list(grid_params.values())

    try:
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    except Exception as e:
        print(f"✗ Error generating grid combinations: {e}")
        traceback.print_exc()
        sys.exit(1)


def sanitize_for_identifier(value: str) -> str:
    """Sanitize a value for use in identifier string."""
    sanitized = str(value)
    # Replace problematic characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


CURRENT_TIME = datetime.now()


def parse_hardcoded_fields(prefix, filename):
    prefix = prefix.replace("$DATETIME0", CURRENT_TIME.strftime("%Y-%m-%d_%H:%M:%S"))
    prefix = prefix.replace("$DATETIME1", CURRENT_TIME.strftime("%Y%m%d-%H:%M:%S"))
    prefix = prefix.replace("$DATETIME", CURRENT_TIME.strftime("%Y%m%d%H%M%S"))
    prefix = prefix.replace("$DATE0", CURRENT_TIME.strftime("%Y-%m-%d"))
    prefix = prefix.replace("$DATE1", CURRENT_TIME.strftime("%Y%m%d"))
    prefix = prefix.replace("$DATE", CURRENT_TIME.strftime("%Y%m%d"))
    prefix = prefix.replace("$FILE", filename)
    return prefix


def generate_identifier(
    params: Dict[str, Any], prefix: str = "", filename: str = ""
) -> str:
    """
    Generate $IDENTIFIER value from parameters.
    Format: prefix_PARAM1_VALUE1_PARAM2_VALUE2_...
    """
    parts = []

    # Add prefix if provided
    if prefix:
        prefix = parse_hardcoded_fields(prefix, filename)
        # prefix = sanitize_for_identifier(prefix)
        parts.append(prefix)

    # Add parameter key-value pairs
    for key, value in sorted(params.items()):
        if key != "name":  # Skip 'name' field
            sanitized_key = sanitize_for_identifier(key)
            sanitized_value = sanitize_for_identifier(str(value))
            parts.append(f"{sanitized_key}_{sanitized_value}")

    return "_".join(parts) if parts else "default"


def apply_replacements(
    template: str,
    replacements: Dict[str, Any],
    identifier: str,
    config_name: Optional[str] = None,
    verbose: bool = False,
    string_delimiter: Optional[str] = '"',
) -> str:
    """Apply replacements to template string using $PLACEHOLDER format."""
    try:
        result = template

        # First, replace $IDENTIFIER
        result = result.replace("$RAWIDENTIFIER", identifier)
        result = result.replace(
            "$IDENTIFIER", string_delimiter + identifier + string_delimiter
        )

        # Then replace all other placeholders
        for key, value in replacements.items():
            if isinstance(value, str):
                str_value = string_delimiter + value + string_delimiter
            else:
                str_value = str(value)
            placeholder = f"${key}"
            result = result.replace(placeholder, str_value)

        if verbose:
            # Check for unreplaced placeholders
            remaining = re.findall(r"\$([A-Z_][A-Z0-9_]*)", result)
            if remaining:
                unique_remaining = set(remaining)
                config_info = f" in config '{config_name}'" if config_name else ""
                print(
                    f"  ⚠ Warning{config_info}: {len(unique_remaining)} placeholder(s) not replaced:"
                )
                for ph in sorted(unique_remaining):
                    print(f"    - ${ph}")

        return result

    except Exception as e:
        config_info = f" for config '{config_name}'" if config_name else ""
        print(f"✗ Error applying replacements{config_info}: {e}")
        traceback.print_exc()
        sys.exit(1)


def sanitize_for_filename(value: str) -> str:
    """Sanitize a value to be safe for use in filenames."""
    sanitized = str(value)
    sanitized = sanitized.replace(".", "_")
    sanitized = sanitized.replace("/", "_")
    sanitized = sanitized.replace("\\", "_")
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace("*", "_")
    sanitized = sanitized.replace("?", "_")
    sanitized = sanitized.replace('"', "_")
    sanitized = sanitized.replace("<", "_")
    sanitized = sanitized.replace(">", "_")
    sanitized = sanitized.replace("|", "_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def generate_filename(
    prefix: str, params: Dict[str, Any], counter: int, config_name: Optional[str] = None
) -> str:
    """Generate a descriptive filename from parameters."""
    parts = [prefix, str(counter)]
    filename_prefix = "_".join(parts)

    # Add config name if provided
    if config_name:
        parts.append(sanitize_for_filename(config_name))

    # Add parameter values
    if params:
        for key, value in params.items():
            if key != "name":  # Skip 'name' field
                sanitized_key = sanitize_for_filename(key)
                sanitized_value = sanitize_for_filename(str(value))
                parts.append(f"{sanitized_key}_{sanitized_value}")

    return filename_prefix, "_".join(parts)


def write_file_safely(path: str, content: str) -> bool:
    """Write file with error handling."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"  ✗ Error writing file '{path}': {e}")
        return False


def main():
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    print_section("Template File Generator with Grid Search & Predefined Configs", "=")

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be created")

    # Display arguments
    print(f"Template file: {args.template}")
    print(f"Config file: {args.config}")

    # Validate inputs exist
    if not os.path.isfile(args.template):
        print(f"✗ Error: Template file '{args.template}' not found!")
        sys.exit(1)

    if not os.path.isfile(args.config):
        print(f"✗ Error: Config file '{args.config}' not found!")
        sys.exit(1)

    # Load configuration and template
    config = load_config(
        args.config,
        args.verbose,
    )
    for key, argval, default in [
        ("id", args.id_prefix, ""),
        ("output_dir", args.output_dir, "experiments"),
        ("output_prefix", args.file_prefix, "exp"),
        ("string_delimiter", args.delimiter, '"'),
    ]:
        if argval is not None:
            print(f"Config key={key} is overwritten with value=<{argval}>!")
            config[key] = argval
        elif key in config:
            print(f"Config key={key} has value=<{config[key]}>.")
        else:
            print(f"Config key={key} is falling back to default=<{default}>!")
            config[key] = default

    template = load_template(args.template, args.verbose)

    # Display configuration summary
    print_section("Configuration Summary")
    print(f"Output directory: {config['output_dir']}")
    print(f"Output prefix: {config['output_prefix']}")
    print(f"Fixed replacements: {len(config['replacements'])} value(s)")
    if config["replacements"]:
        for key, value in config["replacements"].items():
            print(f"  - ${key} = {value}")

    # Determine file extension
    if config["file_extension"]:
        file_extension = config["file_extension"]
        if not file_extension.startswith("."):
            file_extension = "." + file_extension
    else:
        file_extension = os.path.splitext(args.template)[1]

    print(f"Output file extension: {file_extension}")

    # Generate configurations
    all_configs = []

    # Add predefined configs
    if config["predefined_configs"]:
        print_subsection("Predefined Configurations")
        print(f"Found {len(config['predefined_configs'])} predefined configuration(s)")

        for idx, preset in enumerate(config["predefined_configs"], 1):
            config_name = preset.get("name", f"preset_{idx}")
            # Remove 'name' from parameters if present
            params = {k: v for k, v in preset.items() if k != "name"}
            all_configs.append(("predefined", config_name, params))

            identifier = generate_identifier(params, config["id"])
            print(f"  {idx}. {config_name}")
            print(f"     Parameters: {params}")
            print(f"     $IDENTIFIER: {identifier}")

    # Add grid search configs
    if config["grid_search"]:
        print_subsection("Grid Search Configurations")
        print(f"Grid search parameters:")
        for key, values in config["grid_search"].items():
            print(f"  - ${key}: {values} ({len(values)} values)")

        grid_combinations = generate_grid_combinations(config["grid_search"])
        print(f"Total grid combinations: {len(grid_combinations)}")

        for params in grid_combinations:
            all_configs.append(("grid", None, params))

    total_configs = len(all_configs)
    print(f"\nTotal configurations to generate: {total_configs}")

    if args.dry_run:
        print_section("Dry Run Complete")
        print(f"Would generate {total_configs} file(s)")
        print("Run without --dry-run to create files")
        sys.exit(0)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config['output_dir']}_{timestamp}"

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")
        sys.exit(1)

    # Generate files
    print_section("Generating Files")

    generated_files = []
    failed_files = []

    for idx, (config_type, config_name, params) in enumerate(all_configs, 1):
        print(f"\n[{idx}/{total_configs}] Generating configuration...")

        # Generate filename
        filename_prefix, filename = generate_filename(
            config["output_prefix"], params, idx, config_name
        )
        output_path = os.path.join(output_dir, f"{filename}{file_extension}")
        print(f"  Output file: {output_path}")

        # Combine fixed replacements with current config parameters
        all_replacements = {**config["replacements"], **params}

        # Generate identifier
        identifier = generate_identifier(params, config["id"], filename_prefix)

        print(f"  Type: {config_type}")
        if config_name:
            print(f"  Name: {config_name}")
        print(f"  Parameters: {params}")
        print(f"  $IDENTIFIER: {identifier}")

        # Apply replacements to template
        output_content = apply_replacements(
            template,
            all_replacements,
            identifier,
            config_name,
            args.verbose,
            config["string_delimiter"],
        )

        # Write output file
        if write_file_safely(output_path, output_content):
            generated_files.append(
                (output_path, config_type, config_name, params, identifier)
            )
            print(f"  ✓ Successfully created file")
        else:
            failed_files.append(
                (output_path, config_type, config_name, params, identifier)
            )

    # Create execution script
    print_subsection("Creating Execution Script")

    execution_script = f"{config['output_prefix']}_execute_all.sh"

    try:
        with open(execution_script, "w") as es:
            es.write("#!/bin/bash\n")
            es.write(f"# Auto-generated execution script\n")
            es.write(
                f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            es.write(f"# Identifier: '{config["id"]}'\n")
            es.write(f"# Total files: {len(generated_files)}\n\n")

            for (
                file_path,
                config_type,
                config_name,
                params,
                identifier,
            ) in generated_files:
                # Add comment with config info
                if config_name:
                    es.write(f"# Config: {config_name} ({config_type})\n")
                else:
                    es.write(f"# Config: {config_type}\n")
                es.write(f"# Identifier: {identifier}\n")
                es.write(f"# Parameters: {params}\n")

                # Detect file type and add appropriate execution command
                if file_path.endswith(".sbatch"):
                    es.write(f"sbatch {file_path}\n")
                elif file_path.endswith(".sh"):
                    es.write(f"bash {file_path}\n")
                elif file_path.endswith(".py"):
                    es.write(f"python {file_path}\n")
                else:
                    es.write(f"# Execute: {file_path}\n")

                es.write("\n")

        os.chmod(execution_script, 0o755)
        print(f"✓ Created execution script: {execution_script}")

    except Exception as e:
        print(f"✗ Error creating execution script: {e}")
        traceback.print_exc()

    # Final summary
    print_section("Summary")
    print(f"✓ Successfully generated: {len(generated_files)} file(s)")
    if failed_files:
        print(f"✗ Failed to generate: {len(failed_files)} file(s)")
    print(f"Output directory: {output_dir}")
    print(f"Execution script: {execution_script}")

    if generated_files:
        print(f"\nTo execute all files, run:")
        print(f"  bash {execution_script}")

    # Exit with appropriate code
    sys.exit(1 if failed_files else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
