#!/bin/zsh
hidutil property --set '{"UserKeyMapping":[]}'
launchctl bootout "gui/$(id -u)/ai.jev.capslock" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/ai.jev.capslock.plist"
echo "Caps Lock restored."
