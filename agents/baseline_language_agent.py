import random
import time
import os
from civrealm.agents.base_agent import BaseAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.configs import fc_args
from agents.workers.qwen_worker import QwenWorker


MAX_RETRIES = 2  # やり直し回数の上限


class BaselineLanguageAgent(BaseAgent):
    def __init__(self,
                 LLM_model='Lorbus/Qwen3.6-27B-int4-AutoRound',
                 load_dialogue=False):
        super().__init__()

        self.turn = 0
        if "debug.agentseed" in fc_args:
            self.set_agent_seed(fc_args["debug.agentseed"])

        self.qwen_worker = QwenWorker(
            model=LLM_model,
            api_url="http://localhost:8001/v1/chat/completions"
        )
        if load_dialogue:
            self.qwen_worker.load_saved_dialogue()

        # やり直し回数別カウント: {0: 1回目成功, 1: 1回やり直し後成功, 2: 2回やり直し後成功, 'random': 上限超えランダム}
        self.retry_stats = {i: 0 for i in range(MAX_RETRIES + 1)}
        self.retry_stats['random'] = 0

    def interact_with_llm(self, input_prompt, avail_action_list):
        exec_action_name = None
        prompt_addition = ''

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.qwen_worker.generate_command(
                    input_prompt + prompt_addition
                )
                exec_action_name, prompt_addition = self.qwen_worker.process_command(
                    response, input_prompt, avail_action_list
                )
            except Exception as e:
                fc_logger.error(repr(e))

            if exec_action_name is not None:
                self.retry_stats[attempt] += 1
                print(f"\n===== CHOSEN ACTION (attempt {attempt + 1}) =====")
                print(exec_action_name)
                break

        if exec_action_name is None:
            exec_action_name = random.choice(avail_action_list)
            self.retry_stats['random'] += 1
            print(f'retry limit ({MAX_RETRIES}) exceeded, randomly choose: {exec_action_name}')

        return exec_action_name

    def act(self, observations, info):
        available_actions = info['available_actions']
        for ctrl_type in available_actions.keys():
            if ctrl_type == 'unit':
                unit_dict = info['llm_info'][ctrl_type]
                valid_actor_id, valid_actor_name, valid_action_list = \
                    self.get_valid_actor_actions(unit_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                actor_obs  = info['llm_info'][ctrl_type][valid_actor_id]['observations']
                zoom_in_obs  = actor_obs['minimap']
                zoom_out_obs = actor_obs['upper_map']

                obs_input_prompt = self.qwen_worker.prompt_handler.unit_obs_action(
                    ctrl_type=ctrl_type,
                    actor_name=valid_actor_name,
                    zoom_out_obs=zoom_out_obs,
                    zoom_in_obs=zoom_in_obs,
                    available_actions=valid_action_list,
                    general_advise=''
                )

                print('current unit:', valid_actor_name, '; unit id:', valid_actor_id)

                exec_action_name = self.interact_with_llm(
                    obs_input_prompt, valid_action_list
                )

                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

            elif ctrl_type == 'city':
                city_dict = info['llm_info'][ctrl_type]
                valid_actor_id, valid_actor_name, valid_action_list = \
                    self.get_valid_actor_actions(city_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                actor_obs    = info['llm_info'][ctrl_type][valid_actor_id]['observations']
                zoom_in_obs  = actor_obs['minimap']
                zoom_out_obs = actor_obs['upper_map']
                producing    = actor_obs.get('producing', 'NOTHING')

                obs_input_prompt = self.qwen_worker.prompt_handler.city_obs_action(
                    ctrl_type=ctrl_type,
                    actor_name=valid_actor_name,
                    zoom_out_obs=zoom_out_obs,
                    zoom_in_obs=zoom_in_obs,
                    producing=producing,
                    available_actions=valid_action_list,
                    general_advise=''
                )

                print('current city:', valid_actor_name, '; city id:', valid_actor_id)

                exec_action_name = self.interact_with_llm(
                    obs_input_prompt, valid_action_list
                )

                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

        local_time = time.localtime()
        self.qwen_worker.save_dialogue_to_file(
            os.path.join(
                os.getcwd(), "agents/qwen_worker/saved_dialogues/",
                f"saved_dialogue_for_T{info['turn'] + 1:03d}"
                f"_at_{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}.txt"
            ))
        return None

    def get_valid_actor_actions(self, actor_dict, info, ctrl_type):
        if info['turn'] != self.turn:
            self.planned_actor_ids = []
            self.turn = info['turn']

        for actor_id, actor_info in actor_dict.items():
            if actor_id in self.planned_actor_ids:
                continue
            actor_name   = actor_info['name']
            avail_actions = actor_info['available_actions']
            self.planned_actor_ids.append(actor_id)
            return actor_id, actor_name, avail_actions
        return None, None, None
