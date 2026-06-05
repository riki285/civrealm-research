import json
import os
import time


RESULT_TAGS = [
    'Population', 'Trade', 'Production', 'Cities', 'Technologies',
    'Military Service', 'Wonders', 'Research Speed', 'Land Area',
    'Settled Area', 'Literacy', 'Culture', 'Spaceship',
    'Built Units', 'Killed Units', 'Unit Losses', 'Units Used',
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

_CULTURE_IDX   = RESULT_TAGS.index('Culture')
_SPACESHIP_IDX = RESULT_TAGS.index('Spaceship')


def _build_player_record(pid, player_info, game_results, evaluations):
    """player_id ひとつ分のスコア辞書を組み立てる。"""

    def final_val(tag):
        try:
            return evaluations[tag][pid][-1]
        except (KeyError, IndexError, TypeError):
            return None

    gr       = game_results.get(pid, {})
    cat_list = gr.get('category_score', [])

    def cat(idx):
        return cat_list[idx] if isinstance(cat_list, (list, tuple)) and idx < len(cat_list) else None

    return {
        'name':             player_info.get('name'),
        'winner':           gr.get('winner'),
        'score':            gr.get('score') if gr.get('score') is not None else final_val('score'),
        'population':       final_val('population'),
        'researched_techs': final_val('researched_techs'),
        'wonders':          final_val('wonders'),
        'military_units':   final_val('military_units'),
        'gold':             final_val('gold'),
        'cities':           final_val('cities'),
        'culture':          cat(_CULTURE_IDX),
        'spaceship':        cat(_SPACESHIP_IDX),
        'economics':        final_val('economics'),
        'production':       final_val('production'),
        'research_speed':   final_val('research_speed'),
        'land_area':        final_val('land_area'),
        'settled_area':     final_val('settled_area'),
        'units_built':      final_val('units_built'),
        'units_killed':     final_val('units_killed'),
        'units_lost':       final_val('units_lost'),
        'units_used':       final_val('units_used'),
    }


def save_experiment_results(env, agent_name: str = 'BaselineLanguageAgent', extra: dict = None,
                            eval_data=None):
    """ゲーム終了後に全プレイヤーのスコアと内訳を results/ に JSON で保存する。

    eval_data: env.evaluate_game() の戻り値 (players, tags, turns, evaluations)。
               渡さない場合は内部で evaluate_game() を呼ぶ（game_scores/ が追加生成される）。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    game_results = env.get_game_results()
    if eval_data is not None:
        players, _tags, _turns, evaluations = eval_data
    else:
        players, _tags, _turns, evaluations = env.evaluate_game()
    my_player_id = env.unwrapped.civ_controller.player_ctrl.my_player_id

    all_players = {}
    for pid, pinfo in (players or {}).items():
        record = _build_player_record(pid, pinfo, game_results, evaluations or {})
        record['is_llm_agent'] = (pid == my_player_id)
        all_players[str(pid)] = record

    # evaluate_game が None を返したとき (scorelog 未取得) は game_results だけで補完
    if not all_players and game_results:
        for pid, gr in game_results.items():
            all_players[str(pid)] = _build_player_record(pid, {}, game_results, {})
            all_players[str(pid)]['is_llm_agent'] = (pid == my_player_id)

    output = {
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'agent':        agent_name,
        'my_player_id': my_player_id,
        'players':      all_players,
    }
    if extra:
        output.update(extra)

    fname = os.path.join(RESULTS_DIR, f"result_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'[Results saved] {fname}')
    for pid_str, pdata in all_players.items():
        label = '[LLM]' if pdata.get('is_llm_agent') else f'[AI {pid_str}]'
        print(f"  {label} score={pdata.get('score')}, winner={pdata.get('winner')}, "
              f"pop={pdata.get('population')}, techs={pdata.get('researched_techs')}, "
              f"wonders={pdata.get('wonders')}, culture={pdata.get('culture')}, "
              f"spaceship={pdata.get('spaceship')}, military={pdata.get('military_units')}")
    return output


def compose_print(x):
    return f"\033[{x}m"


PRINT_RESUME = compose_print(0)
PRINT_ACTION = compose_print(31) + compose_print(42)
PRINT_STEP = compose_print(1) + compose_print(46)
PRINT_CURRENT = compose_print(4) + compose_print(45)


def print_action(*args):
    print(PRINT_ACTION, *args, PRINT_RESUME)


def print_step(*args):
    print(PRINT_STEP, *args, PRINT_RESUME)


def print_current(*args):
    print(PRINT_CURRENT, *args, PRINT_RESUME)
