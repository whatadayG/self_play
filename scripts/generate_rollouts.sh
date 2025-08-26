/home/nickatomlin/georgiazhou/self_play/.venv/bin/python \
/home/nickatomlin/georgiazhou/self_play/scripts/matching_run_multiple.py \
--exp-name aug24-try-checkpoint1000 \
--resume  \
--end 50 \
--num-gpus 2 \
--user-model-id "$(curl -s http://127.0.0.1:30000/get_model_info | /home/nickatomlin/georgiazhou/self_play/.venv/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model_path") or d.get("model_name") or "")')" \
--agent-model-id "$(curl -s http://127.0.0.1:30000/get_model_info | /home/nickatomlin/georgiazhou/self_play/.venv/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model_path") or d.get("model_name") or "")')" \
--use-sglang True \
--sglang-url http://127.0.0.1:30000/v1 \
--samples-per-game 1
