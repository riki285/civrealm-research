import os
import time

from .mastaba_agent import MastabaAgent
from .workers import QwenMastabaWorker


class QwenMastabaAgent(MastabaAgent):
    """
    MastabaAgent の Qwen ローカル推論版。
    Azure GPT の代わりに vLLM 経由の Qwen3.6-27B-int4-AutoRound を使用する。

    構造:
      - strategy_maker (advisor): ターン開始時に全体戦略を生成
      - workers (controller): エンティティごとに行動を決定（スレッド並列）
    """

    def initialize_workers(self):
        self.strategy_maker = QwenMastabaWorker(role="advisor")
        self.workers = {}

    def add_entity(self, entity_type, entity_id):
        self.workers[(entity_type, entity_id)] = QwenMastabaWorker(
            role="controller",
            ctrl_type=entity_type,
            actor_id=entity_id,
        )

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict, available_actions):
        zoom_in_obs = actor_dict['observations']['minimap']
        zoom_out_obs = actor_dict['observations']['upper_map']

        if ctrl_type == "city":
            producing = actor_dict['observations'].get('producing', "NOTHING")
            available_actions = list(available_actions) + ['produce ' + producing]
            prompt = self.strategy_maker.prompt_handler.city_obs_action(
                ctrl_type=ctrl_type,
                actor_name=actor_name,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                producing=producing,
                available_actions=available_actions,
                general_advise=self.general_advise,
            )
        else:
            prompt = self.strategy_maker.prompt_handler.unit_obs_action(
                ctrl_type=ctrl_type,
                actor_name=actor_name,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                available_actions=available_actions,
                general_advise=self.general_advise,
            )
        return prompt
