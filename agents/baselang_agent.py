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

import os
import time
import threading
from datetime import datetime

from civrealm.freeciv.utils.language_agent_utility import make_action_list_readable, get_action_from_readable_name

from .language_agent import LanguageAgent
from .workers import AzureGPTWorker, QwenWorker
from .utils import print_current, print_action, filter_blacklisted_actions

# Wrong Interpretation of action names. Goto Yexin to fix it.


class BaseLangAgent(LanguageAgent):
    def __init__(self, run_id=None, **kwargs):
        super().__init__(**kwargs)
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = run_id
        self.dialogue_dir = os.path.join(os.getcwd(), 'saved_dialogues', run_id) + '/'
        if not os.path.exists(self.dialogue_dir):
            os.makedirs(self.dialogue_dir)
        self.retry_stats = {}  # {リトライ回数: 決定数, 'random': タイムアウト数}
        self._retry_stats_lock = threading.Lock()

    def initialize_workers(self):
        self.workers = {}

    def add_entity(self, entity_type, entity_id):
        """ self.workers[(entity_type,
                      entity_id)] = AzureGPTWorker(ctrl_type=entity_type,
                                                   actor_id=entity_id) """
        self.workers[(entity_type,
                      entity_id)] = QwenWorker(ctrl_type=entity_type,
                                                   actor_id=entity_id)

    def remove_entity(self, entity_type, entity_id):
        # llm_infoから消えても available_actions にいれば作業中（フィルタされているだけ）
        avail = self.info.get('available_actions', {}).get(entity_type, {})
        if entity_id in avail:
            return  # 生存中なのでワーカーを保持
        del self.workers[(entity_type, entity_id)]

    def process_observations_and_info(self, observations, info):
        self.observations = observations
        self.info = info

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict,
                             available_actions, actor_id=None):
        current_unit_obs = actor_dict['observations']['minimap']
        prev_note = ''
        if actor_id is not None:
            # Fix2: 実際に実行された行動（is_action_valid 通過済み）を参照する
            taken = self.last_taken_actions.get((ctrl_type, actor_id))
            if taken:
                prev_note = f'Your previous action was: {taken[0]}. '
        if ctrl_type == "city":
            producing = actor_dict['observations'].get('producing', "NOTHING")
            prompt = f'{prev_note}The {ctrl_type} is {actor_name}, observation is {current_unit_obs}. The city is producing {producing}. Your available action list is {available_actions}.'
        else:
            prompt = f'{prev_note}The {ctrl_type} is {actor_name}, observation is {current_unit_obs}. Your available action list is {available_actions}.'
        system_message = self.info['llm_info'].get("message", "")
        system_message = ("Game scenario message is: "
                          if system_message else "") + system_message
        return system_message + prompt

    def make_single_decision(self, ctrl_type, actor_id, actor_dict):
        worker = self.workers[(ctrl_type, actor_id)]
        print(type(worker))
        actor_name = actor_dict['name']

        available_actions = actor_dict['available_actions']
        producing = actor_dict['observations'].get('producing', 'NOTHING')

        if 'keep activity' in available_actions:
            available_actions.remove("keep activity")

        if 'cancel order' in available_actions:
            available_actions.remove("cancel order")

        available_actions = filter_blacklisted_actions(available_actions)

        obs_input_prompt = self.get_obs_input_prompt(ctrl_type, actor_name,
                                                     actor_dict,
                                                     available_actions,
                                                     actor_id=actor_id)
        print_current(f'Current {ctrl_type}: {actor_name}')
        print("===== input_prompt =====")
        print(obs_input_prompt)

        print("===== AVAILABLE ACTIONS =====")
        print(available_actions)
        # プロンプトに表示した avail と worker に渡す avail を一致させる
        avail_for_worker = list(available_actions)
        if ctrl_type == "city" and producing and producing != "NOTHING":
            produce_current = 'produce ' + producing
            if produce_current not in avail_for_worker:
                avail_for_worker.append(produce_current)
        exec_action_name = worker.choose_action(obs_input_prompt,
                                                avail_for_worker)
        # retry_stats 集計（複数スレッドから呼ばれるためロック必須）
        rc = getattr(worker, 'last_retry_count', 0)
        with self._retry_stats_lock:
            self.retry_stats[rc] = self.retry_stats.get(rc, 0) + 1

        print_action(f'Action chosen for {actor_name}:', exec_action_name)
        if exec_action_name:
            # Fix1: 都市の生産決定後すぐにキャッシュを更新する（アドバイザーの表示ズレを防ぐ）
            if ctrl_type == "city" and exec_action_name.startswith("produce "):
                cache = getattr(self, '_city_producing_cache', None)
                if cache is not None:
                    cache[actor_id] = exec_action_name[len("produce "):]
        if (exec_action_name and exec_action_name != "produce " + producing):
            self.chosen_actions.put((ctrl_type, actor_id, exec_action_name))

        worker.save_dialogue_to_file(
            os.path.join(
                self.dialogue_dir,
                f"dialogue_T{self.info['turn'] + 1:03d}_{actor_id}_at_{time.strftime('%Y.%m.%d_%H:%M:%S')}.txt"
            ))

    def make_decisions(self):

        threads = []
        for ctrl_type in self.info['llm_info'].keys():
            for actor_id, actor_dict in self.info['llm_info'][ctrl_type].items(
            ):
                thread = threading.Thread(target=self.make_single_decision,
                                          args=(ctrl_type, actor_id,
                                                actor_dict))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()
