import random
import time
import os
from civrealm.agents.base_agent import BaseAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.configs import fc_args
from agents.workers.qwen_worker import QwenWorker


UNIT_RULES = {
    'Settlers': (
        "- Build new cities using 'build city'.\n"
        "- Best spots: Grassland or Plains tile near a River, and near resources"
        " (Wheat, Wine, Fish, Silk, Coal, Whale, Pheasant, Game).\n"
        "- Cities must have minimum distance — do NOT build adjacent to an existing city.\n"
        "- If standing on an existing city tile, move away before building.\n"
        "- Do NOT choose 'join city' when you intend to found a new city.\n"
        "- Use 'upper_map' to find directions with unexplored land."
    ),
    'Workers': (
        "- Improve tiles to boost city production.\n"
        "- Priority 1: Build a road if the current tile has no road yet.\n"
        "- Priority 2: Irrigate if the tile has Wheat, Spice, or other farmable resources.\n"
        "- Priority 3: Mine on Hills or Mountains.\n"
        "- Move toward unimproved tiles near cities when current tile is already developed."
    ),
    'Explorer': (
        "- Reveal the map as quickly as possible.\n"
        "- Always move toward the direction with the most 'tiles unexplored' in 'upper_map'.\n"
        "- IMPORTANT: Ocean tiles are impassable for land units. If 'tile_[direction]_1' shows 'Ocean', do NOT choose 'move [direction]' — pick a different direction.\n"
        "- Avoid 'fortify' or 'keep activity' — keep moving every turn."
    ),
    'default': (
        "- Expand and develop your civilization.\n"
        "- Settlers build cities near Rivers and resources.\n"
        "- Workers build roads and irrigate farmland near cities.\n"
        "- Explorers reveal the map by moving toward unexplored tiles."
    ),
}

CITY_RULES = (
    "- Build Settlers to found new cities and expand your empire.\n"
    "- Build Workers to improve surrounding tiles (roads, irrigation, mines).\n"
    "- Build Granary or other improvements to grow the city faster.\n"
    "- Build military units when threatened by enemies."
)


def get_unit_rules(unit_name: str) -> str:
    for key in UNIT_RULES:
        if key != 'default' and key in unit_name:
            return UNIT_RULES[key]
    return UNIT_RULES['default']


MAX_RETRIES = 2  # やり直し回数の上限


class BaselineLanguageAgent(BaseAgent):
    def __init__(self,
                 LLM_model='Qwen/Qwen2.5-7B-Instruct',
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

                current_unit_name = valid_actor_name
                current_unit_obs  = info['llm_info'][ctrl_type][valid_actor_id]['observations']

                avail_str = "\n".join(f"- {a}" for a in valid_action_list)
                obs_input_prompt = (
                    f"## Rules for {current_unit_name}:\n"
                    f"{get_unit_rules(current_unit_name)}\n\n"
                    f"Unit: {current_unit_name}\n"
                    f"Observation: {current_unit_obs}\n\n"
                    f"AVAILABLE ACTIONS (you MUST choose exactly one of these):\n"
                    f"{avail_str}"
                )

                print('current unit:', current_unit_name, '; unit id:', valid_actor_id)

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

                current_city_name = valid_actor_name
                current_city_obs  = info['llm_info'][ctrl_type][valid_actor_id]['observations']

                avail_str = "\n".join(f"- {a}" for a in valid_action_list)
                obs_input_prompt = (
                    f"## City management rules:\n"
                    f"{CITY_RULES}\n\n"
                    f"City: {current_city_name}\n"
                    f"Observation: {current_city_obs}\n\n"
                    f"AVAILABLE ACTIONS (you MUST choose exactly one of these):\n"
                    f"{avail_str}"
                )

                print('current city:', current_city_name, '; city id:', valid_actor_id)

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
