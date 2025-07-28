#!/bin/zsh
# gnome-terminal --window -e 'tmux new-session -d -s e2e_main_process /bin/zsh -c "/home/nv/px4_policy_deploy_plus/src/run/run_all.sh"'
# tmux new-session -d -s e2e_main_process /bin/zsh -c "/home/nv/px4_policy_deploy_plus/src/run/run_all.sh"
echo "$(date) startup executed" >> /tmp/startup_debug.log

gnome-terminal --window -e 'zsh -c "source /home/nv/px4_policy_deploy_plus/src/run/run_all.sh; exec zsh"'

