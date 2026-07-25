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

import pickle
import warnings
import gymnasium
import civrealm
from datetime import datetime
from agents import BaselineLanguageAgent, AutoGPTAgent, BaseLangAgent, MastabaAgent, QwenMastabaAgent, QwenMastabaRuleAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.configs import fc_args
from civrealm.envs.freeciv_wrapper.llm_wrapper import LLMWrapper
from agents.utils import print_step, print_action, print_current, save_experiment_results
from agents import utils

# FIXME: This is a hack to suppress the warning about the gymnasium spaces. Currently Gymnasium does not support hierarchical actions.
warnings.filterwarnings('ignore',
                        message='.*The obs returned by the .* method.*')


MAX_TURNS = None  # ターン数の上限。None にすると無制限。

SEED = 1  # 初期状況固定用シード
fc_args['debug.randomly_generate_seeds'] = False
fc_args['debug.mapseed'] = SEED
fc_args['debug.gameseed'] = SEED
fc_args['debug.agentseed'] = SEED
fc_args['ruleset'] = 'classic'


def main():
    """
    Main

    Main entry of the program.
    Starts a single-player Freeciv game against rule-based AI.
    """
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = datetime.now()

    env = gymnasium.make('civrealm/FreecivLLM-v0') #CivRealmのGymnasium環境を作成
    #agent = MastabaAgent(max_deconflict_depth=3)
    #agent = BaseLangAgent()
    #agent = BaselineLanguageAgent()
    #agent = QwenMastabaAgent(run_id=run_id, max_deconflict_depth=3)
    agent = QwenMastabaRuleAgent(run_id=run_id, max_deconflict_depth=3)

    observations, info = env.reset()

    done = False
    step = 0
    while not done:
        try:
            action = agent.act(observations, info)
            observations, reward, terminated, truncated, info = env.step(action) #Freecivを１ターン進行
            done = terminated or truncated

            step += 1
            print_step(f'Step: {step}, Turn: {info["turn"]}, ' +
                       f'Reward: {reward}, Terminated: {terminated}, ' +
                       f'Truncated: {truncated}')

            if MAX_TURNS is not None and info['turn'] >= MAX_TURNS:
                print_step(f'Reached turn limit ({MAX_TURNS}). Stopping.')
                break
        except Exception as e:
            fc_logger.error(repr(e))
            raise e
    eval_data = env.evaluate_game()
    env.plot_game_scores()

    model_name = getattr(getattr(agent, 'qwen_worker', None), 'model', None)

    retry_stats = getattr(agent, 'retry_stats', None)
    if retry_stats is not None:
        total = sum(retry_stats.values())
        print(f'\n[Retry stats] total decisions: {total}')
        for k, v in retry_stats.items():
            label = f'{k}回やり直し' if k != 'random' else 'ランダム(上限超え)'
            print(f'  {label}: {v}回 ({v/total*100:.1f}%)' if total > 0 else f'  {label}: {v}回')
    if model_name:
        print(f'  model: {model_name}')

    elapsed = datetime.now() - start_time
    elapsed_sec = int(elapsed.total_seconds())

    save_experiment_results(
        env,
        agent_name=type(agent).__name__,
        extra={
            'model': model_name,
            'retry_stats': retry_stats,
            'elapsed_seconds': elapsed_sec,
            'elapsed_human': f'{elapsed_sec // 3600}h {(elapsed_sec % 3600) // 60}m {elapsed_sec % 60}s',
        },
        eval_data=eval_data,
        run_id=run_id,
    )
    env.close()


if __name__ == '__main__':
    main()
