import json
import os
import re
import time


BLACKLISTED_ACTIONS = frozenset({
    'join city',  # Settlerを消滅させ都市人口+1するだけ — ほぼ常に損
})


def filter_blacklisted_actions(actions):
    """ブラックリストに含まれる行動をリストから除外する。"""
    return [a for a in actions if a not in BLACKLISTED_ACTIONS]


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
                            eval_data=None, run_id: str = None):
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

    fname = os.path.join(RESULTS_DIR, f"result_{run_id or time.strftime('%Y%m%d_%H%M%S')}.json")
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


# ---------------------------------------------------------------------------
# Observation grid formatters
# ---------------------------------------------------------------------------

_MINIMAP_GRID = [
    ['tile_north_2_west_2', 'tile_north_2_west_1', 'tile_north_2',  'tile_north_2_east_1', 'tile_north_2_east_2'],
    ['tile_north_1_west_2', 'tile_north_1_west_1', 'tile_north_1',  'tile_north_1_east_1', 'tile_north_1_east_2'],
    ['tile_west_2',         'tile_west_1',         'current_tile',  'tile_east_1',         'tile_east_2'        ],
    ['tile_south_1_west_2', 'tile_south_1_west_1', 'tile_south_1',  'tile_south_1_east_1', 'tile_south_1_east_2'],
    ['tile_south_2_west_2', 'tile_south_2_west_1', 'tile_south_2',  'tile_south_2_east_1', 'tile_south_2_east_2'],
]
_MINIMAP_LABELS = {
    'tile_north_2_west_2': 'N2W2', 'tile_north_2_west_1': 'N2W1', 'tile_north_2': 'N2',  'tile_north_2_east_1': 'N2E1', 'tile_north_2_east_2': 'N2E2',
    'tile_north_1_west_2': 'N1W2', 'tile_north_1_west_1': 'N1W1', 'tile_north_1': 'N1',  'tile_north_1_east_1': 'N1E1', 'tile_north_1_east_2': 'N1E2',
    'tile_west_2':         'W2',   'tile_west_1':         'W1',   'current_tile': '[YOU]','tile_east_1':         'E1',   'tile_east_2':         'E2',
    'tile_south_1_west_2': 'S1W2', 'tile_south_1_west_1': 'S1W1', 'tile_south_1': 'S1',  'tile_south_1_east_1': 'S1E1', 'tile_south_1_east_2': 'S1E2',
    'tile_south_2_west_2': 'S2W2', 'tile_south_2_west_1': 'S2W1', 'tile_south_2': 'S2',  'tile_south_2_east_1': 'S2E1', 'tile_south_2_east_2': 'S2E2',
}

_UPPER_GRID = [
    ['block_north_1_west_1', 'block_north_1', 'block_north_1_east_1'],
    ['block_west_1',         'current_block', 'block_east_1'        ],
    ['block_south_1_west_1', 'block_south_1', 'block_south_1_east_1'],
]
_UPPER_LABELS = {
    'block_north_1_west_1': 'NW', 'block_north_1': 'N',     'block_north_1_east_1': 'NE',
    'block_west_1':         'W',  'current_block': '[YOU]', 'block_east_1':         'E',
    'block_south_1_west_1': 'SW', 'block_south_1': 'S',     'block_south_1_east_1': 'SE',
}


def _fmt_tile(values):
    """Convert a tile value list into a compact single-line string."""
    if not values:
        return '---'
    parts = []
    ownership = None
    for v in values:
        v = str(v)
        if 'tiles unexplored' in v:
            return '???'
        if 'unit owners are:' in v:
            rest = v.split('unit owners are:')[1].strip()
            ownership = 'me' if 'myself' in rest else (rest if rest else '?')
            continue
        m = re.match(r'^(\d+) (.+)$', v)
        if m:
            n, name = int(m.group(1)), m.group(2)
            if 'cities of myself' in v:
                parts.append('City(me)')
            elif 'cities of' in v:
                owner = v.split('cities of')[1].strip()
                parts.append(f'City({owner})' if owner else 'City(?)')
            elif n == 1:
                parts.append(name)
            else:
                parts.append(f'{n}x{name}')
        else:
            parts.append(v)
    if ownership:
        parts.append(f'[{ownership}]')
    return ', '.join(parts) if parts else '---'


def _fmt_block(values):
    """Summarise a block's value list for the upper_map grid."""
    if not values:
        return '---'
    unexplored = 0
    terrain_counts = []
    resources = []
    units = []
    cities = []
    own_flag = False
    enemy_names = []
    for v in values:
        v = str(v)
        if v.startswith('OWNCITY:'):
            cities.append(v[8:])  # 都市名付き（city_name_at 経由）、正規表現より先にチェック
            continue
        m = re.match(r'^(\d+) (.+)$', v)
        if not m:
            continue
        n, name = int(m.group(1)), m.group(2)
        if 'tiles unexplored' in v:
            unexplored = n
        elif 'unit owners are:' in v:
            if 'myself' in v:
                own_flag = True
            else:
                rest = v.split('unit owners are:')[1].strip()
                if rest and rest not in enemy_names:
                    enemy_names.append(rest)
        elif 'cities of myself' in v:
            cities.append(f'City(me)x{n}' if n > 1 else 'City(me)')
        elif 'cities of' in v:
            rest = v.split('cities of')[1].strip()
            label = rest if rest else '?'
            cities.append(f'City({label})x{n}' if n > 1 else f'City({label})')
        elif name in ('Road', 'Mine', 'Irrigation', 'Fortress', 'River'):
            resources.append(f'{name}x{n}' if n > 1 else name)
        elif name in ('Plains', 'Grassland', 'Hills', 'Mountains', 'Forest',
                      'Desert', 'Ocean', 'Deep Ocean', 'Swamp', 'Lake',
                      'Tundra', 'Arctic'):
            terrain_counts.append((n, name))
        else:
            resources.append(f'{name}x{n}' if n > 1 else name)

    parts = []
    if unexplored:
        parts.append(f'?x{unexplored}')
    terrain_counts.sort(key=lambda x: -x[0])
    parts.extend(f'{name}x{n}' if n > 1 else name for n, name in terrain_counts[:3])
    parts.extend(resources[:3])
    parts.extend(cities)
    if own_flag:
        parts.append('[me]')
    for ename in enemy_names:
        parts.append(f'[{ename}]')
    return ', '.join(parts) if parts else '---'


def _cell(text, w):
    return f' {text:<{w}} '


def _split_lines(text, w):
    """Split text into lines of at most w chars, breaking at commas."""
    if len(text) <= w:
        return [text]
    parts = text.split(', ')
    lines, cur = [], ''
    for p in parts:
        if cur and len(cur) + 2 + len(p) > w:
            lines.append(cur)
            cur = p
        else:
            cur = (cur + ', ' + p) if cur else p
    if cur:
        lines.append(cur)
    return lines or [text]


def _render_grid(grid_keys, label_map, data, fmt_fn, cell_w):
    """Render data as an ASCII grid table with multi-line cells (no truncation)."""
    cols = len(grid_keys[0])
    sep = '+' + '+'.join(['-' * (cell_w + 2)] * cols) + '+'
    out = [sep]
    for row_keys in grid_keys:
        cell_contents = [
            (label_map.get(k, k), _split_lines(fmt_fn(data.get(k)), cell_w))
            for k in row_keys
        ]
        max_h = max(len(lines) for _, lines in cell_contents)
        out.append('|' + '|'.join(_cell(lbl, cell_w) for lbl, _ in cell_contents) + '|')
        for i in range(max_h):
            out.append('|' + '|'.join(
                _cell(lines[i] if i < len(lines) else '', cell_w)
                for _, lines in cell_contents
            ) + '|')
        out.append(sep)
    return '\n'.join(out)


# タイルキー → アクターからの (dx, dy) オフセット (x増→East, y増→South)
_TILE_KEY_TO_OFFSET = {
    'current_tile':          ( 0,  0),
    'tile_north_1':          ( 0, -1), 'tile_south_1':          ( 0,  1),
    'tile_east_1':           ( 1,  0), 'tile_west_1':           (-1,  0),
    'tile_north_1_east_1':   ( 1, -1), 'tile_north_1_west_1':   (-1, -1),
    'tile_south_1_east_1':   ( 1,  1), 'tile_south_1_west_1':   (-1,  1),
    'tile_north_2':          ( 0, -2),
    'tile_north_2_east_1':   ( 1, -2), 'tile_north_2_west_1':   (-1, -2),
    'tile_north_2_east_2':   ( 2, -2), 'tile_north_2_west_2':   (-2, -2),
    'tile_south_2':          ( 0,  2),
    'tile_south_2_east_1':   ( 1,  2), 'tile_south_2_west_1':   (-1,  2),
    'tile_south_2_east_2':   ( 2,  2), 'tile_south_2_west_2':   (-2,  2),
    'tile_east_2':           ( 2,  0), 'tile_west_2':           (-2,  0),
    'tile_north_1_east_2':   ( 2, -1), 'tile_north_1_west_2':   (-2, -1),
    'tile_south_1_east_2':   ( 2,  1), 'tile_south_1_west_2':   (-2,  1),
}


def format_minimap_grid(minimap, city_name_at=None):
    """Format minimap dict as a 5x5 ASCII grid ([YOU] = unit position).

    city_name_at: optional dict {(dx, dy): city_name} — own cities in range
                  will be labelled with their name instead of 'City(me)'.
    """
    if not isinstance(minimap, dict):
        return str(minimap)

    if city_name_at:
        minimap = dict(minimap)  # shallow copy — don't mutate caller's data
        for tile_key, offset in _TILE_KEY_TO_OFFSET.items():
            if offset not in city_name_at or tile_key not in minimap:
                continue
            city_name = city_name_at[offset]
            minimap[tile_key] = [
                city_name if 'cities of myself' in str(v) else v
                for v in minimap[tile_key]
            ]

    return _render_grid(_MINIMAP_GRID, _MINIMAP_LABELS, minimap, _fmt_tile, cell_w=20)


# ブロック方向 (bx, by) → ブロックキー
# 各ブロックの中心は actor から (bx*5, by*5) タイル離れた位置 (block_radius=2 → block_size=5)
_BLOCK_SIZE = 5
_BLOCK_HALF = _BLOCK_SIZE // 2  # = 2
_BLOCK_DIR_TO_KEY = {
    ( 0,  0): 'current_block',
    ( 0, -1): 'block_north_1',    ( 0,  1): 'block_south_1',
    ( 1,  0): 'block_east_1',     (-1,  0): 'block_west_1',
    ( 1, -1): 'block_north_1_east_1', (-1, -1): 'block_north_1_west_1',
    ( 1,  1): 'block_south_1_east_1', (-1,  1): 'block_south_1_west_1',
}


def format_upper_map_grid(upper_map, city_name_at=None):
    """Format upper_map dict as a 3x3 ASCII grid ([YOU] = current block).

    city_name_at: optional dict {(dx, dy): city_name} (same format as
                  format_minimap_grid). Own cities will be labelled with
                  their name instead of 'City(me)'.
    """
    if not isinstance(upper_map, dict):
        return str(upper_map)

    if city_name_at:
        # 各都市がどのブロックに属するかを計算し、そのブロックの値リストを書き換える
        block_cities = {}  # block_key → [city_name, ...]
        for (dx, dy), city_name in city_name_at.items():
            bx = by = None
            for b in (-1, 0, 1):
                if abs(dx - b * _BLOCK_SIZE) <= _BLOCK_HALF:
                    bx = b
                    break
            for b in (-1, 0, 1):
                if abs(dy - b * _BLOCK_SIZE) <= _BLOCK_HALF:
                    by = b
                    break
            if bx is None or by is None:
                continue
            block_key = _BLOCK_DIR_TO_KEY.get((bx, by))
            if block_key:
                block_cities.setdefault(block_key, []).append(city_name)

        upper_map = dict(upper_map)
        for block_key, city_names in block_cities.items():
            if block_key not in upper_map:
                continue
            name_iter = iter(city_names)
            new_vals = []
            for v in upper_map[block_key]:
                if 'cities of myself' in str(v):
                    # "N cities of myself player_X" → "OWNCITY:{city_name}"
                    # _fmt_block の OWNCITY: 分岐で cities リストに直接追加される
                    new_vals.append(f'OWNCITY:{next(name_iter, "City(me)")}')
                else:
                    new_vals.append(v)
            # 置換しきれなかった都市名（同ブロックに複数の都市）を末尾に追加
            for remaining in name_iter:
                new_vals.append(f'OWNCITY:{remaining}')
            upper_map[block_key] = new_vals

    return _render_grid(_UPPER_GRID, _UPPER_LABELS, upper_map, _fmt_block, cell_w=30)


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
