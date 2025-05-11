#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_QUADRUPEDS_DIR="$SCRIPT_DIR"
TASKS_DIR="$RL_QUADRUPEDS_DIR/tasks"
ASSETS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_assets"
REWARDS_DIR="$RL_QUADRUPEDS_DIR/quadrupeds_rewards"

# Default values
PIP_CMD="/home/ltoschi/Documents/IsaacLab/isaaclab.sh -p -m pip"
ACTION=""
SHOW_HELP=false

# Print help message
print_help() {
    echo "Usage: $(basename "$0") [--pip | --isaaclab] [install | uninstall]"
    echo
    echo "Options:"
    echo "  --pip           Use the system 'pip'"
    echo "  --isaaclab      Use 'isaaclab -p -m pip' (default)"
    echo "  install         Install all packages"
    echo "  uninstall       Uninstall all packages"
    echo "  -h, --help      Show this help message"
    echo
    echo "Example:"
    echo "  ./script.sh --pip install"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pip) PIP_CMD="pip" ;;
        --isaaclab) PIP_CMD="/home/ltoschi/Documents/IsaacLab/isaaclab.sh -p -m pip" ;;
        install) ACTION="install" ;;
        uninstall) ACTION="uninstall" ;;
        -h|--help) SHOW_HELP=true ;;
        *) echo "Unknown option: $1"; SHOW_HELP=true ;;
    esac
    shift
done

# Show help if no action or help flag is set
if [[ "$SHOW_HELP" = true || -z "$ACTION" ]]; then
    print_help
    return 0 2>/dev/null || exit 0
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

echo "${ACTION^}ing packages in tasks directory..."
for dir in "$TASKS_DIR"/*/; do
    process_package "$dir" "$(basename "$dir")"
done