# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import time
import json
import os
from civrealm.agents.base_agent import BaseAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger
# from civrealm.freeciv.utils.language_agent_utility import MOVE_NAMES, INVERSE_MOVE_NAMES
from civrealm.configs import fc_args
#from agents.civ_autogpt import GPTAgent
from agents.workers.qwen_worker import QwenWorker


MOVE_NAMES = {'goto_0': 'move_NorthWest', 'goto_1': 'move_North', 'goto_2': 'move_NorthEast',
              'goto_3': 'move_West', 'goto_4': 'move_East', 'goto_5': 'move_SouthWest',
              'goto_6': 'move_South', 'goto_7': 'move_SouthEast'}
INVERSE_MOVE_NAMES = {val: key for key, val in MOVE_NAMES.items()}


class BaselineLanguageAgent(BaseAgent):
    def __init__(self,
                 LLM_model='Qwen/Qwen2.5-7B-Instruct',
                 load_dialogue=False):
        super().__init__()

        self.turn = 0
        if "debug.agentseed" in fc_args: #乱数固定(実験再現性)
            self.set_agent_seed(fc_args["debug.agentseed"])

        #self.gpt_agent = GPTAgent(model=LLM_model)
        self.qwen_worker = QwenWorker(
            model=LLM_model,
            api_url="http://localhost:8001/v1/chat/completions"
        )
        if load_dialogue:
            self.qwen_worker.load_saved_dialogue()

    def interact_with_llm_within_time_limit(self,
                                            input_prompt,
                                            current_ctrl_obj_name,
                                            avail_action_list,
                                            interact_timeout=120):
        exec_action_name = None
        start_time = time.time()
        while exec_action_name is None:
            end_time = time.time()
            if (end_time - start_time) >= interact_timeout: #タイムアウト処理
                exec_action_name = random.choice(avail_action_list)
                #self.gpt_agent.dialogue.pop(-1)
                #self.gpt_agent.dialogue.pop(-1)
                print('overtime, randomly choose:', exec_action_name)
                break
            try:
                print("\n===== PROMPT TO QWEN =====")
                print(input_prompt)

                response = self.qwen_worker.generate_command(
                    input_prompt
                )

                print("\n===== RESPONSE =====")
                print(response)

                exec_action_name, retry_prompt = \
                    self.qwen_worker.process_command(
                        response,
                        input_prompt,
                        avail_action_list
                    )

                print("\n===== CHOSEN ACTION =====")
                print(exec_action_name)

            except Exception as e:
                fc_logger.error(repr(e))
                continue
                # response = self.gpt_agent.communicate(input_prompt,
                #                                       parse_choice_tag=False)
                # self.gpt_agent.memory.save_context(
                #     {'user': input_prompt}, {'assistant': str(response)})
                # exec_action_name = self.gpt_agent.process_command(
                #     response, input_prompt, current_ctrl_obj_name,
                #     avail_action_list)
            # except Exception as e:
            #     fc_logger.error('Error in interact_with_llm_within_time_limit')
            #     fc_logger.error(repr(e))
            #     fc_logger.error('input_prompt: ' + input_prompt)
            #     continue
        return exec_action_name

    def act(self, observations, info):
        available_actions = info['available_actions']
        for ctrl_type in available_actions.keys():
            if ctrl_type == 'unit':
                unit_dict = info['llm_info'][ctrl_type] #LLM用に整形済みの情報
                fc_logger.debug(f'unit_dict: {unit_dict}')
                #行動するユニットの選択
                valid_actor_id, valid_actor_name, valid_action_list = self.get_valid_actor_actions(
                    unit_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                current_unit_name = valid_actor_name
                current_unit_obs = info['llm_info'][ctrl_type][valid_actor_id]['observations']
                fc_logger.debug(f'current unit obs: {current_unit_obs}')

                # current_avail_actions_list = [
                #     MOVE_NAMES[action_name]
                #     if action_name in MOVE_NAMES.keys() else action_name
                #     for action_name in valid_action_list
                # ]
                current_avail_actions_list = valid_action_list

                #unitにおける入力プロンプト
                obs_input_prompt = f"""
                You are a AI that helps me playing freeciv web game.
                I will give the tile information and your available actions.
                please chose the best action from action list according to the instruction.
                Your answer should be given in the format: {{"action":"ACTION_NAME"}} and I only need on line.

                Instruction: Focus on cities on fertile Grassland-River tiles. Develop production on Plains, explore Forests for resources, and assess Mountains and Hills for defense. Manage resources and engage diplomatically for prosperity.

                The unit is {current_unit_name}, observation is {current_unit_obs}. Your available action list is {current_avail_actions_list}. 
                """
                
                print('current unit:', current_unit_name, '; unit id:',valid_actor_id)

                exec_action_name = self.interact_with_llm_within_time_limit(
                    obs_input_prompt, current_unit_name,
                    current_avail_actions_list)

                # try:
                #     exec_action_name = INVERSE_MOVE_NAMES[exec_action_name]
                # except:
                #     pass
                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

            elif ctrl_type == 'city':
                city_dict = info['llm_info'][ctrl_type]
                fc_logger.debug(f'city_dict: {city_dict}')
                #行動する都市の選択
                valid_actor_id, valid_actor_name, valid_action_list = self.get_valid_actor_actions(city_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                current_city_name = valid_actor_name
                current_city_obs = info['llm_info'][ctrl_type][valid_actor_id]['observations']
                fc_logger.debug(f'current city obs: {current_city_obs}')

                current_avail_actions_list = valid_action_list

                #cityにおける入力プロンプト
                obs_input_prompt = f"""
                You are a AI that helps me playing freeciv web game.
                I will give the tile information and your available actions.
                please chose the best action from action list according to the instruction.
                Your answer should be given in the format: {{"action":"ACTION_NAME"}} and I only need on line.

                Instruction: Focus on cities on fertile Grassland-River tiles. Develop production on Plains, explore Forests for resources, and assess Mountains and Hills for defense. Manage resources and engage diplomatically for prosperity.
                
                The city is {current_city_name}, observation is {current_city_obs}. Your available action list is {current_avail_actions_list}. 
                """
                
                print('current city:', current_city_name, '; city id:',
                      valid_actor_id)

                exec_action_name = self.interact_with_llm_within_time_limit(
                    obs_input_prompt, current_city_name,
                    current_avail_actions_list)

                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

            else:
                continue

        local_time = time.localtime()
        self.qwen_worker.save_dialogue_to_file(
            os.path.join(
                os.getcwd(), "agents/qwen_worker/saved_dialogues/" +
                f"saved_dialogue_for_T{info['turn'] + 1:03d}_at_{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}.txt"
            ))
        # self.gpt_agent.save_dialogue_to_file(
        #     os.path.join(
        #         os.getcwd(), "agents/civ_autogpt/saved_dialogues/" +
        #         f"saved_dialogue_for_T{info['turn'] + 1:03d}_at_{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}.txt"
        #     ))
        return None

    def get_valid_actor_actions(self, actor_dict, info, ctrl_type):#今ターンに行動していないunitを選択

        if info['turn'] != self.turn:
            self.planned_actor_ids = []
            self.turn = info['turn']

        for actor_id, actor_info in actor_dict.items():
            if actor_id in self.planned_actor_ids:
                continue
            
            actor_name = actor_info['name']
            avail_actions = actor_info['available_actions']
            self.planned_actor_ids.append(actor_id)
            return actor_id, actor_name, avail_actions
        return None, None, None
