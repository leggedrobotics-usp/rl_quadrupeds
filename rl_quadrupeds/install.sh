#!/bin/bash
shopt -s expand_aliases

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_QUADRUPEDS_DIR="$SCRIPT_DIR"
TASKS_DIR="$RL_QUADRUPEDS_DIR/tasks"
ASSETS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_assets"
REWARDS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_rewards"
COMMANDS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_commands"
OBSERVATIONS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_observations"
ISAACLAB_EXT_DIR="$RL_QUADRUPEDS_DIR/isaaclab_extensions"

# Default values
PIP_CMD="pip"
ACTION=""
SHOW_HELP=false

# Print help message
print_help() {
    echo "Usage: $(basename "$0") [--pip | --isaaclab] [install | uninstall]"
    echo
    echo "Options:"
    echo "  install         Install all packages"
    echo "  uninstall       Uninstall all packages"
    echo "  --pip           Use the system 'pip' (default)"
    echo "  --isaaclab      Use 'isaaclab -p -m pip'"
    echo "  -h, --help      Show this help message"
    echo
    echo "Example:"
    echo "  ./script.sh --pip install"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pip)
            PIP_CMD="pip"
            ;;
        --isaaclab)
            if [[ -z "$ISAACLAB_FOLDER" ]]; then
                echo "Error: ISAACLAB_FOLDER environment variable is not defined."
                echo "Please set it to the IsaacLab root directory and try again."
                exit 1
            fi
            PIP_CMD="$ISAACLAB_FOLDER/isaaclab.sh -p -m pip"
            ;;
        install)
            ACTION="install"
            ;;
        uninstall)
            ACTION="uninstall"
            ;;
        -h|--help)
            SHOW_HELP=true
            ;;
        *)
            echo "Unknown option: $1"
            SHOW_HELP=true
            ;;
    esac
    shift
done


# Show help if requested
if [[ "$SHOW_HELP" = true ]]; then
    print_help
    return 0 2>/dev/null || exit 0
fi

# If no action is specified, warn and exit without closing terminal
if [[ -z "$ACTION" ]]; then
    echo "Error: No action specified (install or uninstall)."
    echo "Use -h or --help for usage information."
    return 1 2>/dev/null || exit 1
fi

echo "Using pip command: $PIP_CMD"
echo "Action: $ACTION"
echo "Target directory: $RL_QUADRUPEDS_DIR"

process_package() {
    local dir="$1"
    local name="$2"
    if [[ -f "$dir/setup.py" ]]; then
        echo "${ACTION^}ing $name..."
        if [[ "$ACTION" == "install" ]]; then
            $PIP_CMD install --force-reinstall --no-cache-dir -e "$dir"
        elif [[ "$ACTION" == "uninstall" ]]; then
            PACKAGE_NAME=$(basename "$dir")
            $PIP_CMD uninstall -y "$PACKAGE_NAME"
        fi
    else
        echo "Skipping $name — no setup.py found."
    fi
}

# Process each package
process_package "$ASSETS_DIR" "quadrupeds_assets"
process_package "$REWARDS_DIR" "quadrupeds_rewards"
process_package "$COMMANDS_DIR" "quadrupeds_commands"
process_package "$OBSERVATIONS_DIR" "quadrupeds_observations"
process_package "$ISAACLAB_EXT_DIR" "isaaclab_extensions"

echo "${ACTION^}ing packages in tasks directory..."
for dir in "$TASKS_DIR"/*/; do
    process_package "$dir" "$(basename "$dir")"
done