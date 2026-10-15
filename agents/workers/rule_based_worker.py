"""
RuleBasedWorker — advisor の指示に従って行動するルールベース controller

設計方針:
    advisor が「何をするか」を決め、worker は「どう実行するか」だけを決める。
    worker は自分で戦略を立てない。

4層アーキテクチャ:
    第1層 意図解釈       advisor テキストから自分宛ての意図を抽出
    第2層 実行可能性判定  その行動が今この場で選べるか照合（表記ゆれ吸収）
    第3層 戦術判断       Freeciv AI 由来の評価で実行方法を決定
    第4層 フォールバック  判断不能時の安全な既定行動

安全弁（案C: 破滅的行動のみ拒否）:
    advisor の指示は原則そのまま実行するが、以下だけは拒否して代替行動を取る。
      - 勝率が SUICIDE_WIN_CHANCE 未満の攻撃
      - 食料が枯渇する立地への都市建設
    拒否は必ず self.safety_rejections に記録し、実験の評価指標として集計する。

移植元 (github.com/freeciv/freeciv):
    unit_win_chance()      common/combat.c        → win_chance()
    assess_danger()        ai/default/daimilitary.c:709  → _assess_danger()
    cityresult_fill()      ai/default/daisettler.c:275   → evaluate_city_site()
    food_starvation()      ai/default/daisettler.c:569   → CitySiteResult.starving
    naval_bonus()          ai/default/daisettler.c:625   → NAVAL_EMPHASIS
    autoworkers.c                                        → _tactic_improve()

使い方 (QwenMastabaAgent._create_controller_worker):
    return RuleBasedWorker(ctrl_type=entity_type, actor_id=entity_id, agent=self)
"""

import re
import math
import random


# ============================================================
# 定数
# ============================================================

# --- 立地評価の重み ---
FOOD_WEIGHT = 3       # 成長は序盤の最重要指標
SHIELD_WEIGHT = 2
TRADE_WEIGHT = 1

CENTER_WEIGHT = 2.0   # 都市中心タイルは必ず使われるので重み付け
WORKABLE_TILES = 8    # 小都市が実際に働けるおおよそのタイル数
NAVAL_EMPHASIS = 20   # 海隣接ボーナス(%)  daisettler.c:112
RIVER_BONUS_PCT = 15  # 河川ボーナス(%)
FOOD_COST = 2         # game.info.food_cost（市民1人あたりの食料消費）
OVERLAP_PENALTY = 12  # 半径2以内の自都市1つあたりの減点

# 閾値の根拠（実測スコア）:
#   優良地（草原+河川+小麦+海隣接）      91
#   良好地（草原+丘石炭+小麦、公式例）    54  ← ここは建てるべき
#   標準地（草原に草原）                 40
#   自都市に隣接（重複あり）             28  ← 建てさせたくない
#   ツンドラ・砂漠                    14-18  ← 飢餓判定で却下
SITE_ENOUGH = 50      # これ以上なら即建設      (RESULT_IS_ENOUGH 相当)
SITE_ACCEPTABLE = 32  # 彷徨い上限に達したら妥協して建設する下限
MAX_WANDER_TURNS = 5  # 適地を探し続ける上限ターン数 (daisettler.c:84 の思想)
MAX_WANDER_HARD = 15  # この回数を超えたら強制建設 or 既存都市から強制脱出

# --- 安全弁 ---
SUICIDE_WIN_CHANCE = 0.10   # これ未満の勝率の攻撃は拒否する

# --- 戦闘計算 (Freeciv) ---
SHIELD_WEIGHTING = 17       # ai/default/daiunit.h


# ============================================================
# classic ruleset の地形産出 (food, shield, trade)
# ============================================================
TERRAIN_OUTPUT = {
    'grassland': (2, 0, 0),
    'plains':    (1, 1, 0),
    'forest':    (1, 2, 0),
    'hills':     (1, 0, 0),
    'mountains': (0, 1, 0),
    'desert':    (0, 1, 0),
    'tundra':    (1, 0, 0),
    'swamp':     (1, 0, 0),
    'jungle':    (1, 0, 0),
    'ocean':     (1, 0, 2),
    'deep ocean': (0, 0, 2),
    'lake':      (2, 0, 2),
    'glacier':   (0, 0, 0),
    'arctic':    (0, 0, 0),
}

# 特産物・地形特性のボーナス (food, shield, trade)
EXTRA_BONUS = {
    'wheat':     (2, 0, 0),
    'buffalo':   (0, 2, 0),
    'oasis':     (3, 0, 0),
    'oil':       (0, 3, 0),
    'resources': (0, 2, 0),
    'coal':      (0, 2, 0),
    'wine':      (0, 0, 4),
    'gold':      (0, 0, 6),
    'iron':      (0, 3, 0),
    'game':      (2, 0, 0),
    'pheasant':  (2, 0, 0),
    'silk':      (0, 0, 3),
    'furs':      (0, 0, 3),
    'peat':      (0, 4, 0),
    'spice':     (0, 0, 3),
    'fruit':     (3, 0, 0),
    'gems':      (0, 0, 4),
    'fish':      (3, 0, 0),
    'whales':    (2, 2, 0),
    'ivory':     (0, 0, 4),
    'seals':     (3, 0, 0),
    'musk ox':   (2, 0, 0),
    'river':     (0, 0, 1),
}

OCEAN_TERRAINS = frozenset({'ocean', 'deep ocean', 'lake'})


# ============================================================
# ユニット性能表 (attack, defense, hp, firepower, build_cost)
# ============================================================
UNIT_STATS = {
    'warriors':   (1, 1, 10, 1, 10),
    'phalanx':    (1, 2, 10, 1, 20),
    'archers':    (3, 2, 10, 1, 30),
    'legion':     (4, 2, 10, 1, 40),
    'pikemen':    (1, 2, 10, 1, 20),
    'musketeers': (3, 3, 20, 1, 30),
    'riflemen':   (5, 4, 20, 1, 40),
    'mech. inf.': (6, 6, 30, 1, 50),
    'horsemen':   (2, 1, 10, 1, 20),
    'chariot':    (3, 1, 10, 1, 30),
    'knights':    (4, 2, 10, 1, 40),
    'dragoons':   (5, 2, 20, 1, 50),
    'cavalry':    (8, 3, 20, 1, 60),
    'armor':      (10, 5, 30, 1, 80),
    'catapult':   (6, 1, 10, 1, 40),
    'cannon':     (8, 1, 20, 1, 40),
    'artillery':  (10, 1, 20, 2, 50),
    'settlers':   (0, 1, 20, 1, 30),
    'engineers':  (0, 2, 20, 1, 40),
    'workers':    (0, 1, 10, 1, 30),
    'explorer':   (0, 1, 10, 1, 30),
    'diplomat':   (0, 1, 10, 1, 30),
    'caravan':    (0, 1, 10, 1, 50),
}
_DEFAULT_STATS = (1, 1, 10, 1, 10)

# advisor が品目を明示しないときの代替候補（強い順）
DEFENDER_PRIORITY = ['Mech. Inf.', 'Riflemen', 'Musketeers',
                     'Pikemen', 'Phalanx', 'Warriors']
ATTACKER_PRIORITY = ['Armor', 'Cavalry', 'Dragoons', 'Knights',
                     'Legion', 'Archers', 'Horsemen', 'Warriors']

SETTLER_KINDS = frozenset({'settlers', 'engineers'})
WORKER_KINDS = frozenset({'workers', 'engineers'})
EXPLORER_KINDS = frozenset({'explorer'})


# ============================================================
# 方向
# ============================================================
DIR_VECTORS = {
    'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0),
    'northeast': (1, -1), 'northwest': (-1, -1),
    'southeast': (1, 1), 'southwest': (-1, 1),
}
# 行動名として使われる表記（小文字・キャメル両方に対応するため正規化して照合する）
DIR_NAMES = ['north', 'south', 'east', 'west',
             'northeast', 'northwest', 'southeast', 'southwest']

# Freecivの goto_N 形式（数字）→ 方向名マッピング
# 配置: 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE
GOTO_NUM_TO_DIR = {
    '0': 'northwest', '1': 'north', '2': 'northeast',
    '3': 'west',                    '4': 'east',
    '5': 'southwest', '6': 'south', '7': 'southeast',
}


# ============================================================
# 意図
# ============================================================
class Intent:
    BUILD_CITY = 'BUILD_CITY'
    EXPLORE = 'EXPLORE'
    DEFEND = 'DEFEND'
    ATTACK = 'ATTACK'
    IMPROVE = 'IMPROVE'
    PRODUCE = 'PRODUCE'
    UNKNOWN = 'UNKNOWN'


# advisor テキストからの意図検出キーワード（優先順に評価する）
INTENT_KEYWORDS = [
    (Intent.BUILD_CITY, ['build city', 'build cities', 'found a city',
                         'found cities', 'settle', 'establish cit']),
    (Intent.ATTACK,     ['attack', 'kill', 'destroy', 'eliminate',
                         'assault', 'offensive', 'strike']),
    (Intent.DEFEND,     ['defend', 'defence', 'defense', 'fortify',
                         'protect', 'garrison', 'guard',
                         'escort', 'accompany', 'patrol']),
    (Intent.EXPLORE,    ['explore', 'scout', 'unexplored', 'reveal']),
    (Intent.IMPROVE,    ['irrigat', 'mine', 'road', 'improve terrain',
                         'develop terrain', 'terrain improvement']),
]

# advisor の文がどのエンティティ種別に宛てられているか
ENTITY_ALIASES = {
    'settlers':  ['settler', 'settlers'],
    'workers':   ['worker', 'workers'],
    'explorer':  ['explorer', 'explorers', 'scout'],
    'city':      ['city', 'cities'],
    'military':  ['military', 'soldier', 'troop', 'army', 'unit',
                  'warrior', 'warriors', 'phalanx', 'legion', 'archers',
                  'musketeers', 'riflemen', 'pikemen', 'horsemen',
                  'knights', 'cavalry', 'armor', 'defender'],
}


# _extract_item で品目候補から除外する語（都市参照・地形名など）
_NON_ITEM_WORDS = frozenset({
    'city', 'cities', 'capital',
    'plains', 'grassland', 'forest', 'hill', 'hills',
    'mountain', 'mountains', 'ocean', 'sea', 'river',
    'coast', 'desert', 'jungle', 'tundra', 'swamp',
})


# ============================================================
# 汎用ヘルパー
# ============================================================
def _norm(s):
    """小文字化・アンダースコアをスペースに統一・空白を1つに潰す。行動名の表記ゆれ吸収用。"""
    s = re.sub(r'_', ' ', (s or '').strip().lower())
    return re.sub(r'\s+', ' ', s)


def win_chance(att_str, att_hp, att_fp, def_str, def_hp, def_fp):
    """
    Freeciv の unit_win_chance() (common/combat.c) を再現する。

    Freeciv の戦闘はラウンド制:
      各ラウンドで攻撃側が att/(att+def) の確率で勝ち、敗者は相手の
      firepower 分 HP を失う。HP が 0 になった方が負け。
    したがって勝率は「n_lose 回負ける前に n_win 回勝つ確率」= 負の二項分布。

    単純な戦力比では勝率を大きく誤る。
    例: Legion(攻4) vs Warriors(防1) は戦力比だと 80% だが、
        10ラウンド戦うので実際の勝率は約 99.8%。
    """
    if att_str <= 0:
        return 0.0
    if def_str <= 0:
        return 1.0

    p = att_str / (att_str + def_str)
    n_win = max(1, math.ceil(def_hp / max(1, att_fp)))
    n_lose = max(1, math.ceil(att_hp / max(1, def_fp)))

    if n_win > 200 or n_lose > 200:
        return p

    total = 0.0
    for k in range(n_lose):
        total += math.comb(n_win - 1 + k, k) * (p ** n_win) * ((1 - p) ** k)
    return min(1.0, total)


def _obs_num(raw, key, default=None):
    """観測辞書（numpy配列の場合あり）から int を安全に取り出す。"""
    if raw is None:
        return default
    val = raw.get(key)
    if val is None:
        return default
    try:
        return int(val.flat[0]) if hasattr(val, 'flat') else int(val)
    except Exception:
        return default


# ============================================================
# 観測パーサー
#
# agents/utils.py の _render_grid() が生成する ASCII テーブルを逆変換する。
# 実際の出力形式（agents/qwen_mastaba_agent.py:422 が生成）:
#
#   +----------------------+----------------------+----------------------+
#   | N1W1                 | N1                   | N1E1                 |
#   | ---                  | Hills, Coal          | Mountains, Legion    |
#   |                      |                      | [player_1]           |
#   +----------------------+----------------------+----------------------+
#   | W1                   | [YOU]                | E1                   |
#   | Forest, River        | Grassland            | Grassland, Roma      |
#   +----------------------+----------------------+----------------------+
#
# セル内容の規則（utils.py:151 _fmt_tile / utils.py:184 _fmt_block）:
#   '---'            空タイル
#   '???'            未探索（minimap）
#   '?xN'            未探索 N マス（upper_map）
#   'City(me)'       自都市（city_name_at 未指定時）
#   'City(player_1)' 他プレイヤーの都市
#   'Roma'           自都市（city_name_at 指定時は実名に置換される）
#   '2xLegion'       同種ユニット2体
#   '[me]'           このタイルのユニットは自軍
#   '[player_1]'     このタイルのユニットは player_1 のもの
#   'Grasslandx8'    ブロック内の地形数（upper_map のみ）
#
# 長いセルは _split_lines() でカンマ区切りで折り返されるため、
# 複数行を結合してから解釈する必要がある。
# ============================================================

# ラベル → オフセット（x増→East, y増→South）
# utils.py の _MINIMAP_LABELS / _UPPER_LABELS に対応する
_LABEL_NS_RE = re.compile(r'([NS])(\d*)')
_LABEL_EW_RE = re.compile(r'([EW])(\d*)')


def _label_to_offset(label):
    """
    グリッドのラベルを (dx, dy) に変換する。
      '[YOU]' -> (0, 0)      'N2W1' -> (-1, -2)
      'E1'    -> (1, 0)      'NW'   -> (-1, -1)   （upper_map は数字なし）
    """
    lb = (label or '').strip().upper()
    if lb in ('[YOU]', 'YOU', ''):
        return (0, 0)
    if not re.fullmatch(r'(?:[NS]\d*)?(?:[EW]\d*)?', lb):
        return None

    dx = dy = 0
    m = _LABEL_NS_RE.search(lb)
    if m:
        n = int(m.group(2)) if m.group(2) else 1
        dy = -n if m.group(1) == 'N' else n
    m = _LABEL_EW_RE.search(lb)
    if m:
        n = int(m.group(2)) if m.group(2) else 1
        dx = -n if m.group(1) == 'W' else n
    return (dx, dy)


def _split_grid_cells(text):
    """
    ASCII グリッドを {(dx, dy): cell_text} に変換する。

    _render_grid() の構造:
        セパレータ行 '+---+---+'
        ラベル行     '| N1 | N2 |'
        内容行 (1行以上、折り返しの場合は複数行)
        セパレータ行
    """
    cells = {}
    if not text:
        return cells

    lines = [ln for ln in text.splitlines() if ln.strip()]
    pending_labels = None
    pending_rows = []

    def flush():
        if not pending_labels:
            return
        for idx, label in enumerate(pending_labels):
            offset = _label_to_offset(label)
            if offset is None:
                continue
            parts = [row[idx].strip() for row in pending_rows
                     if idx < len(row) and row[idx].strip()]
            cells[offset] = ', '.join(parts)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('+') and set(stripped) <= set('+-'):
            flush()
            pending_labels = None
            pending_rows = []
            continue
        if not stripped.startswith('|'):
            continue

        # '| a | b |' -> ['a', 'b']
        columns = [c.strip() for c in stripped.strip('|').split('|')]
        if pending_labels is None:
            # 最初の行はラベル行。全列がラベルとして解釈できるか確認する
            if all(_label_to_offset(c) is not None for c in columns):
                pending_labels = columns
            continue
        pending_rows.append(columns)

    flush()
    return cells


# セル内トークンの分類用
_QTY_RE = re.compile(r'^(\d+)x(.+)$')          # '2xLegion' / 'Grasslandx8' は別扱い
_BLOCK_QTY_RE = re.compile(r'^(.+?)x(\d+)$')    # 'Grasslandx8' -> ('Grassland', 8)
_OWNER_RE = re.compile(r'^\[(.+)\]$')          # '[me]' / '[player_1]'
_CITY_RE = re.compile(r'^City\((.+?)\)(?:x(\d+))?$')

# ユニット名（所有者判定に使う）— UNIT_STATS のキーで判定するため定数は不要

_KNOWN_TILE_WORDS = frozenset({'road', 'mine', 'irrigation', 'fortress',
                               'railroad', 'farmland', 'buoy', 'ruins'})


class TileInfo:
    """グリッド1マス分の情報。"""

    __slots__ = ('dx', 'dy', 'terrain', 'extras', 'own_city', 'enemy_city',
                 'city_name', 'own_units', 'enemy_units', 'hut', 'unexplored',
                 'terrain_counts')

    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
        self.terrain = None
        self.extras = []
        self.own_city = False
        self.enemy_city = False
        self.city_name = None
        self.own_units = []
        self.enemy_units = []
        self.hut = False
        self.unexplored = 0
        self.terrain_counts = {}     # upper_map 用 {地形名: マス数}

    @property
    def distance(self):
        """チェビシェフ距離（斜め移動を1とする）。"""
        return max(abs(self.dx), abs(self.dy))

    @property
    def is_ocean(self):
        return self.terrain in OCEAN_TERRAINS

    @property
    def is_empty(self):
        return self.terrain is None and not self.terrain_counts

    @property
    def has_enemy(self):
        return bool(self.enemy_units) or self.enemy_city

    def output(self):
        """(food, shield, trade) を返す。"""
        f, s, t = TERRAIN_OUTPUT.get(self.terrain, (0, 0, 0))
        for ex in self.extras:
            bf, bs, bt = EXTRA_BONUS.get(ex, (0, 0, 0))
            f += bf
            s += bs
            t += bt
        return f, s, t

    def weighted(self):
        f, s, t = self.output()
        return f * FOOD_WEIGHT + s * SHIELD_WEIGHT + t * TRADE_WEIGHT

    def block_quality(self):
        """upper_map ブロックの地形品質を返す。"""
        total = 0.0
        for name, n in self.terrain_counts.items():
            f, s, t = TERRAIN_OUTPUT.get(name, (0, 0, 0))
            total += n * (f * FOOD_WEIGHT + s * SHIELD_WEIGHT + t * TRADE_WEIGHT)
        return total

    def __repr__(self):
        return (f'<Tile({self.dx},{self.dy}) {self.terrain} {self.extras} '
                f'own_city={self.own_city} enemy={self.enemy_units}>')


def _parse_cell(dx, dy, cell_text, is_block=False):
    """1セル分のテキストを TileInfo に変換する。"""
    tile = TileInfo(dx, dy)
    text = (cell_text or '').strip()
    if not text or text == '---':
        return tile

    # 未探索マーカー
    if text == '???':
        tile.unexplored = 1
        return tile

    owner_tag = None
    unit_tokens = []

    for raw in text.split(','):
        token = raw.strip()
        if not token:
            continue

        # 所有者タグ '[me]' / '[player_1]'
        m = _OWNER_RE.match(token)
        if m:
            owner_tag = m.group(1).strip().lower()
            continue

        # 都市 'City(me)' / 'City(player_1)' / 'City(me)x2'
        m = _CITY_RE.match(token)
        if m:
            owner = m.group(1).strip().lower()
            if owner in ('me', 'myself'):
                tile.own_city = True
            else:
                tile.enemy_city = True
            continue

        # 未探索（ブロック） '?x4'
        if token.startswith('?x'):
            try:
                tile.unexplored = int(token[2:])
            except ValueError:
                tile.unexplored = 1
            continue
        if token == '???':
            tile.unexplored = max(tile.unexplored, 1)
            continue

        low = _norm(token)

        # 'Nx地形' 形式（ブロック集計） 'Grasslandx8'
        if is_block:
            m = _BLOCK_QTY_RE.match(token)
            if m:
                name = _norm(m.group(1))
                try:
                    n = int(m.group(2))
                except ValueError:
                    n = 1
                if name in TERRAIN_OUTPUT:
                    tile.terrain_counts[name] = tile.terrain_counts.get(name, 0) + n
                    continue
                if name in EXTRA_BONUS:
                    tile.extras.append(name)
                    continue
                unit_tokens.extend([name] * n)
                continue
            if low in TERRAIN_OUTPUT:
                tile.terrain_counts[low] = tile.terrain_counts.get(low, 0) + 1
                continue

        # 'NxUnit' 形式（ミニマップ） '2xLegion'
        m = _QTY_RE.match(token)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                n = 1
            name = _norm(m.group(2))
            if name in TERRAIN_OUTPUT:
                tile.terrain = tile.terrain or name
            elif name in EXTRA_BONUS:
                tile.extras.append(name)
            else:
                unit_tokens.extend([name] * n)
            continue

        # 村（hut）
        if 'minor tribe village' in low or low == 'hut':
            tile.hut = True
            continue

        if low in TERRAIN_OUTPUT:
            tile.terrain = tile.terrain or low
            continue
        if low in EXTRA_BONUS:
            tile.extras.append(low)
            continue
        if low in _KNOWN_TILE_WORDS:
            tile.extras.append(low)
            continue
        if low in UNIT_STATS:
            unit_tokens.append(low)
            continue

        # 上記いずれでもない語の扱い。
        # _fmt_tile() は値リストの順に出力するため地形が先頭に来る。したがって:
        #   地形が未確定 → ルールセット差異による未知の地形とみなす
        #                  (TERRAIN_OUTPUT に無いので産出は 0 として扱われる)
        #   地形が確定済 → utils.py:317 で 'City(me)' が置換された自都市の実名
        if tile.terrain is None and not is_block:
            tile.terrain = low
            continue
        tile.own_city = True
        tile.city_name = token

    # ユニットの所有者を確定する
    if unit_tokens:
        if owner_tag in ('me', 'myself', None):
            tile.own_units = unit_tokens
        else:
            tile.enemy_units = unit_tokens

    return tile


_ZOOM_IN_RE = re.compile(
    r'zoomed-in observation is\s*(.*?)(?=\nThis city is|\nOur cities|'
    r'\nYour position|\nThe available actions|\Z)', re.S | re.I)
_ZOOM_OUT_RE = re.compile(
    r'zoomed-out observation is\s*(.*?)(?=\nThe zoomed-in|\Z)', re.S | re.I)


def parse_zoom_in(prompt):
    """
    zoomed-in observation（5x5 ASCII グリッド）を {(dx,dy): TileInfo} に変換する。
    グリッドが見つからない場合は空辞書を返す。
    """
    m = _ZOOM_IN_RE.search(prompt or '')
    if not m:
        return {}
    cells = _split_grid_cells(m.group(1))
    return {offset: _parse_cell(offset[0], offset[1], text, is_block=False)
            for offset, text in cells.items()}


def parse_zoom_out(prompt):
    """
    zoomed-out observation（3x3 ASCII グリッド）を {(dx,dy): TileInfo} に変換する。
    各セルはブロック（5x5タイル）の集計値を持つ。
    """
    m = _ZOOM_OUT_RE.search(prompt or '')
    if not m:
        return {}
    cells = _split_grid_cells(m.group(1))
    return {offset: _parse_cell(offset[0], offset[1], text, is_block=True)
            for offset, text in cells.items()}


# ============================================================
# 都市立地評価 — daisettler.c の cityresult 移植
# ============================================================
class CitySiteResult:
    __slots__ = ('score', 'starving', 'no_shield', 'detail')

    def __init__(self, score, starving, no_shield, detail):
        self.score = score
        self.starving = starving
        self.no_shield = no_shield
        self.detail = detail

    @property
    def viable(self):
        return not (self.starving or self.no_shield)


def evaluate_city_site(tiles, center=(0, 0)):
    """
    zoomed-in observation のタイル群から都市立地を評価する。

    Freeciv の cityresult_fill() (daisettler.c:275) に対応:
        total = 中心タイルの産出 + 周辺タイルの産出
              + 海隣接ボーナス + 河川ボーナス
              - 自都市重複ペナルティ

    却下条件 (daisettler.c:569, 584):
        food_starvation()   : 中心 + 最良タイルの食料 <= food_cost
        shield_starvation() : 中心 + 最良タイルのシールド == 0

    Returns: CitySiteResult
    """
    cx, cy = center
    center_tile = tiles.get(center)
    if center_tile is None or center_tile.terrain is None:
        return CitySiteResult(0.0, True, True, 'no terrain data')
    if center_tile.is_ocean:
        return CitySiteResult(0.0, True, True, 'ocean tile')

    # --- 中心タイル ---
    cf, cs, ct = center_tile.output()
    # 都市中心は無償の改善を受ける（最低1シールド・1交易）
    cs = max(cs, 1)
    ct = max(ct, 1)
    center_w = cf * FOOD_WEIGHT + cs * SHIELD_WEIGHT + ct * TRADE_WEIGHT

    # --- 周辺タイル（中心からの相対座標で半径2以内） ---
    others = []
    ocean_adjacent = False
    own_cities_near = 0
    for (dx, dy), tile in tiles.items():
        if (dx, dy) == center or tile.terrain is None:
            continue
        dist = max(abs(dx - cx), abs(dy - cy))
        if dist > 2:
            continue
        if dist <= 1 and tile.is_ocean:
            ocean_adjacent = True
        if tile.own_city:
            own_cities_near += 1
        others.append(tile)

    others.sort(key=lambda t: t.weighted(), reverse=True)
    best_other = others[0] if others else None
    work_sum = sum(t.weighted() for t in others[:WORKABLE_TILES])

    # --- 飢餓判定 (food_starvation) ---
    bf, bs, bt = best_other.output() if best_other else (0, 0, 0)
    starving = (cf + bf) <= FOOD_COST
    no_shield = (cs + bs) == 0

    # --- スコア合成 ---
    score = center_w * CENTER_WEIGHT + work_sum

    if ocean_adjacent:
        score *= 1.0 + NAVAL_EMPHASIS / 100.0
    if 'river' in center_tile.extras:
        score *= 1.0 + RIVER_BONUS_PCT / 100.0

    score -= own_cities_near * OVERLAP_PENALTY
    score = max(0.0, score)

    detail = (f'food={cf}+{bf} shield={cs}+{bs} '
              f'center_w={center_w:.0f} work={work_sum:.0f} '
              f'ocean={"y" if ocean_adjacent else "n"} '
              f'river={"y" if "river" in center_tile.extras else "n"} '
              f'overlap={own_cities_near}')

    return CitySiteResult(score, starving, no_shield, detail)


# ============================================================
# RuleBasedWorker
# ============================================================
class RuleBasedWorker:
    """
    advisor の指示に従って行動を選ぶルールベース controller。
    BaseLangAgent.make_single_decision() と互換のインターフェースを持つ。
    """

    def __init__(self, ctrl_type='unit', actor_id=-1, agent=None):
        self.ctrl_type = ctrl_type
        self.actor_id = actor_id
        self.agent = agent

        # BaseLangAgent の retry_stats 集計用（LLM を使わないので常に 0）
        self.last_retry_count = 0

        # --- ターンをまたいで保持する状態 ---
        self.wander_turns = 0            # 開拓者が適地を探し続けたターン数
        self.last_wander_offset = None   # 前ターンの移動方向（振動防止用）
        self.is_fortified_on_city = False  # 都市上でfortify済みフラグ（翌ターン離散防止）
        self.safety_rejections = []      # 安全弁が発動した記録

        # --- 直近の判断の記録（ログ用） ---
        self.trace = {}

    # ------------------------------------------------------------
    # BaseWorker 互換スタブ
    # ------------------------------------------------------------
    def save_dialogue_to_file(self, save_path):
        """LLM の対話ログの代わりに判断過程を書き出す。"""
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f'entity       : {self.ctrl_type} {self.actor_id}\n')
                for key in ('advisor_text', 'intent', 'feasibility',
                            'tactical', 'safety', 'decision', 'avail'):
                    if key in self.trace:
                        f.write(f'{key:13s}: {self.trace[key]}\n')
                if self.safety_rejections:
                    f.write(f'rejections   : {len(self.safety_rejections)}\n')
                    for r in self.safety_rejections[-5:]:
                        f.write(f'  - {r}\n')
        except Exception:
            pass

    # ============================================================
    # メインエントリ
    # ============================================================
    def choose_action(self, input_prompt, avail_action_list):
        avail = [a for a in (avail_action_list or []) if a]
        if not avail:
            return None

        self.trace = {}
        self.trace['avail'] = avail
        prompt = input_prompt or ''

        try:
            # --- 第1層: 意図解釈 ---
            advisor_text = self._extract_advisor_text(prompt)
            kind = self._entity_kind(prompt)
            intent, target = self._parse_intent(advisor_text, kind)
            self.trace['advisor_text'] = advisor_text[:200] or '(none)'
            self.trace['intent'] = f'{intent} target={target} kind={kind}'

            # --- 第2〜3層: 実行 ---
            action = self._execute(intent, target, kind, prompt, avail)
        except Exception as e:
            self.trace['tactical'] = f'exception: {e!r}'
            action = None

        # --- 第4層: フォールバック ---
        if action is None or action not in avail:
            if action is not None:
                self.trace['feasibility'] = f'"{action}" not in avail -> fallback'
            action = self._fallback(avail, prompt)
            self.trace.setdefault('tactical', 'fallback')

        self.trace['decision'] = action
        return action

    # ============================================================
    # 第1層: 意図解釈
    # ============================================================
    @staticmethod
    def _extract_advisor_text(prompt):
        """
        advisor からの指示を取り出す。

        実際のテンプレート (prompt_collections/civ_prompts_qwen/*.txt):
            Message from advisor: <% general_advise %>
        BaseLang 系のテンプレートでは 'From advisor:' 表記なので両方に対応する。
        """
        m = re.search(r'(?:message\s+)?from advisor\s*:\s*(.*?)(?:\n\s*\n|$)',
                      prompt or '', re.S | re.I)
        return m.group(1).strip() if m else ''

    def _entity_kind(self, prompt):
        """
        自分が何者かを返す。
        'city' / 'settlers' / 'workers' / 'explorer' / 'military' / 'other'
        """
        if self.ctrl_type == 'city':
            return 'city'
        m = re.search(r'controlling\s+\w+\s*:\s*([\w.\s]+?)\s*\d*\s*\.',
                      prompt or '', re.I)
        raw = _norm(m.group(1)) if m else ''
        # "Mech. Inf. 12" のような名前から末尾の ID を除く
        name = re.sub(r'\s*\d+$', '', raw).strip()
        self.trace['unit_type'] = name or '(unknown)'

        if name in SETTLER_KINDS:
            return 'settlers'
        if name in WORKER_KINDS:
            return 'workers'
        if name in EXPLORER_KINDS:
            return 'explorer'
        stats = UNIT_STATS.get(name, _DEFAULT_STATS)
        return 'military' if stats[0] > 0 else 'other'

    def _parse_intent(self, advisor_text, kind):
        """
        advisor テキストから自分宛ての意図を抽出する。

        advisor は "[全体状況] + [種別ごとの指示]" の形で出力する:
          "We are in peace. So explorer should explore unexplored area.
           Settlers should build cities near extras. Cities should build more Settlers."

        Returns: (intent, target_item)
        """
        if not advisor_text:
            return self._default_intent(kind, advisor_text), None

        aliases = ENTITY_ALIASES.get(kind, [])
        sentences = re.split(r'(?<=[.!?;])\s+|\n', advisor_text)

        # 1) 自分の種別名を含む文を優先。
        #    都市の場合: 'capital'/'switch production'/'produce'/'change production' を
        #    含む文を一般的な 'city' 言及より先に処理する（Settler 指示に含まれる
        #    'build city' に誤マッチするのを防ぐため）。
        if kind == 'city':
            capital_sentences = []
            city_sentences = []
            for sent in sentences:
                low = _norm(sent)
                if re.search(r'\b(capital|switch\s+production|change\s+production)\b', low):
                    capital_sentences.append(sent)
                elif any(re.search(r'\b' + re.escape(a) + r'\b', low) for a in aliases):
                    city_sentences.append(sent)
            matched = capital_sentences + city_sentences
        else:
            matched = []
            for sent in sentences:
                low = _norm(sent)
                if any(re.search(r'\b' + re.escape(a) + r'\b', low) for a in aliases):
                    matched.append(sent)

        # 2) 該当がなければ全文から探す（種別を明示しない指示に対応）
        pool = matched if matched else sentences

        for sent in pool:
            low = _norm(sent)
            for intent, keywords in INTENT_KEYWORDS:
                if any(k in low for k in keywords):
                    if intent == Intent.BUILD_CITY and kind != 'settlers':
                        # "cities should build X" は都市の生産指示
                        if kind == 'city':
                            return Intent.PRODUCE, self._extract_item(sent)
                        continue
                    if kind == 'city' and intent in (Intent.DEFEND, Intent.ATTACK):
                        # 都市に対する軍事指示は「軍事ユニットを作れ」の意味
                        return Intent.PRODUCE, self._extract_item(sent) or (
                            '__defender__' if intent == Intent.DEFEND
                            else '__attacker__')
                    return intent, self._extract_item(sent)

            # 生産指示（"build more Settlers" / "produce Granary" / "switch production to X"）
            if kind == 'city':
                item = self._extract_item(sent)
                if item and re.search(r'\b(build|produce|make|construct|switch)\b', low):
                    return Intent.PRODUCE, item

        return self._default_intent(kind, advisor_text), None

    @staticmethod
    def _extract_item(sentence):
        """
        文中から生産品目・対象名を取り出す。
        "Cities should build more Settlers" → "Settlers"
        大文字始まりの語を品目候補とみなす（Freeciv の品目名は固有名詞）。
        ALL-CAPS の語（NOW, SWITCH 等の強調語）は品目名ではないので除外する。
        都市参照語（City 168 等）や地形名（Plains 等）は _NON_ITEM_WORDS で除外する。
        """
        if not sentence:
            return None
        # "switch production to X" の明示パターンを最優先
        m = re.search(r'\bswitch\s+production\s+to\s+([A-Za-z][A-Za-z\s.]+?)(?:\s+(?:now|immediately|asap)\b|[,;!.]|$)',
                      sentence, re.I)
        if m:
            # 冠詞 (a/an/the) を先頭から除去
            item = re.sub(r'^(a|an|the)\s+', '', m.group(1).strip(), flags=re.I)
            # "Worker or Archer" → "Worker"（最初の選択肢だけ使う）
            item = re.split(r'\s+or\s+', item, maxsplit=1, flags=re.I)[0].strip()
            return item or None
        # 複数語の品目名（"City Walls" 等）を文全体から探す（"City" が除外対象のため先にチェック）
        for multi in ('City Walls', 'Mech. Inf.', 'Great Wall'):
            if multi.lower() in sentence.lower():
                return multi
        # 文頭の語は主語なので除外する
        words = sentence.strip().split()
        candidates = []
        for i, w in enumerate(words[1:], start=1):
            clean = w.strip('.,;:!?"\'()')
            if not clean:
                continue
            # 先頭大文字かつ全大文字でない
            if not (clean[0].isupper() and len(clean) > 2 and not clean.isupper()):
                continue
            # 都市参照語・地形名を除外（"Plains/Grassland" のようなスラッシュ区切りも対象）
            parts = re.split(r'[/\\]', clean)
            if all(p.lower() in _NON_ITEM_WORDS for p in parts if p):
                continue
            # 直後が数字なら都市ID参照 ("City 168" の "City") → スキップ
            if i < len(words) - 1 and words[i + 1].strip('.,;:!?"\'()').isdigit():
                continue
            candidates.append(clean)
        if not candidates:
            return None
        return candidates[-1]

    @staticmethod
    def _default_intent(kind, advisor_text):
        """自分宛ての指示が読み取れないときの既定意図。"""
        at_war = bool(re.search(r'\b(at war|in war|under attack|enemy|barbarian|pirate)\b',
                                _norm(advisor_text)))
        if kind == 'settlers':
            return Intent.BUILD_CITY
        if kind == 'workers':
            return Intent.IMPROVE
        if kind == 'explorer':
            return Intent.EXPLORE
        if kind == 'military':
            return Intent.DEFEND
        if kind == 'city':
            return Intent.PRODUCE
        return Intent.EXPLORE

    # ============================================================
    # 第2層: 実行可能性判定（表記ゆれ吸収）
    # ============================================================
    @staticmethod
    def _find_exact(avail, name):
        """正規化した完全一致で行動を探す。"""
        target = _norm(name)
        for a in avail:
            if _norm(a) == target:
                return a
        return None

    @staticmethod
    def _find_contains(avail, *fragments):
        """すべての断片を含む行動を探す（最初の1件）。"""
        frags = [_norm(f) for f in fragments]
        for a in avail:
            low = _norm(a)
            if all(f in low for f in frags):
                return a
        return None

    @staticmethod
    def _find_all_starting(avail, prefix):
        p = _norm(prefix)
        return [a for a in avail if _norm(a).startswith(p)]

    def _find_move(self, avail, direction):
        """
        方向名から移動アクションを探す。
        'move north' / 'move North' / 'goto_N' / 'goto_1'（数字）などの表記ゆれを吸収する。
        """
        d = _norm(direction)
        for a in avail:
            low = _norm(a)
            # goto_数字 形式（例: goto_1 → 'goto 1'）への対応
            m = re.match(r'^goto (\d)$', low)
            if m:
                dir_name = GOTO_NUM_TO_DIR.get(m.group(1), '')
                if dir_name == d:
                    return a
                continue
            if not (low.startswith('move') or low.startswith('goto')):
                continue
            tail = re.sub(r'^(move|goto)\s*', '', low)
            if tail == d:
                return a
        return None

    def _available_moves(self, avail):
        """利用可能な移動を {(dx,dy): action} で返す。"""
        moves = {}
        for name in DIR_NAMES:
            action = self._find_move(avail, name)
            if action:
                moves[DIR_VECTORS[name]] = action
        return moves

    def _find_produce(self, avail, item):
        """
        生産アクションを探す。以下すべての表記に対応する:
          'produce Settlers'
          'change unit prod Settlers 0'
          'change improve prod Barracks 3'
        """
        if not item:
            return None
        low_item = _norm(item)
        for a in avail:
            low = _norm(a)
            if low_item not in low:
                continue
            if 'prod' in low or low.startswith('build'):
                return a
        return None

    def _all_produce_actions(self, avail):
        return [a for a in avail
                if 'prod' in _norm(a) or _norm(a).startswith('produce')]

    # ============================================================
    # 第3層: 戦術判断（意図ごとのディスパッチ）
    # ============================================================
    def _execute(self, intent, target, kind, prompt, avail):
        if intent == Intent.BUILD_CITY:
            return self._tactic_build_city(prompt, avail)
        if intent == Intent.ATTACK:
            return self._tactic_attack(prompt, avail, kind)
        if intent == Intent.DEFEND:
            return self._tactic_defend(prompt, avail)
        if intent == Intent.IMPROVE:
            return self._tactic_improve(prompt, avail)
        if intent == Intent.EXPLORE:
            return self._tactic_explore(prompt, avail)
        if intent == Intent.PRODUCE:
            return self._tactic_produce(target, prompt, avail)
        return None

    # ------------------------------------------------------------
    # BUILD_CITY — 立地評価して建設 or 移動
    # ------------------------------------------------------------
    def _tactic_build_city(self, prompt, avail):
        tiles = parse_zoom_in(prompt)
        build_action = (self._find_exact(avail, 'build city')
                        or self._find_contains(avail, 'build', 'city')
                        or self._find_contains(avail, 'found', 'city'))

        result = evaluate_city_site(tiles, (0, 0))
        self.trace['tactical'] = (f'site_score={result.score:.0f} '
                                  f'[{result.detail}] '
                                  f'wander={self.wander_turns}/{MAX_WANDER_TURNS}')

        if build_action is None:
            # ここには建てられない（海上・既存都市に近すぎ等）→ 適地を探す
            self.wander_turns += 1
            self.trace['feasibility'] = 'build city not available'
            # ハードリミット超過: 都市密集地から強制脱出
            if self.wander_turns >= MAX_WANDER_HARD:
                escape = self._escape_from_cities(tiles, prompt, avail)
                if escape:
                    self.trace['tactical'] += (f' | hard limit({self.wander_turns})'
                                               f' -> escape from cities')
                    return escape
            return self._move_to_best_site(tiles, prompt, avail)

        self.trace['feasibility'] = f'build available: "{build_action}"'

        # --- 安全弁: 食料が枯渇する立地は拒否する ---
        if not result.viable:
            reason = 'food starvation' if result.starving else 'no shield'
            self._reject(f'build city rejected ({reason}): {result.detail}')
            self.wander_turns += 1
            move = self._move_to_best_site(tiles, prompt, avail)
            if move:
                return move
            # 移動もできないなら指示どおり建てる（袋小路の回避）
            self.trace['safety'] += ' | no escape, building anyway'
            self.wander_turns = 0
            self.last_wander_offset = None
            return build_action

        # --- 立地の質で判断 ---
        if result.score >= SITE_ENOUGH:
            self.wander_turns = 0
            self.last_wander_offset = None
            return build_action

        if self.wander_turns >= MAX_WANDER_TURNS and result.score >= SITE_ACCEPTABLE:
            self.trace['tactical'] += ' | wander limit -> compromise'
            self.wander_turns = 0
            self.last_wander_offset = None
            return build_action

        # ハードリミット: スコアに関わらず強制建設
        if self.wander_turns >= MAX_WANDER_HARD:
            self.trace['tactical'] += f' | hard limit({self.wander_turns}) -> force build'
            self.wander_turns = 0
            self.last_wander_offset = None
            return build_action

        # まだ良い場所を探す
        self.wander_turns += 1
        move = self._move_to_best_site(tiles, prompt, avail)
        if move:
            return move

        # 移動不能なら建てる
        self.trace['tactical'] += ' | cannot move -> build here'
        self.wander_turns = 0
        self.last_wander_offset = None
        return build_action

    def _move_to_best_site(self, tiles, prompt, avail):
        """
        隣接タイルを都市中心候補として評価し、最良の方向へ移動する。
        半径2の観測しかないため候補ごとの情報は部分的だが、相対比較には使える。
        判定できない場合は zoomed-out のブロック品質で方角を決める。
        """
        moves = self._available_moves(avail)
        if not moves:
            return None

        scored = []
        for (dx, dy), action in moves.items():
            tile = tiles.get((dx, dy))
            if tile is not None and (tile.is_ocean or tile.own_city):
                continue
            res = evaluate_city_site(tiles, (dx, dy))
            if not res.viable:
                continue
            scored.append((res.score, action, (dx, dy)))

        if scored:
            scored.sort(reverse=True, key=lambda x: x[0])
            # 前ターンと逆方向への移動は振動になるので2番手以降を優先
            reverse = ((-self.last_wander_offset[0], -self.last_wander_offset[1])
                       if self.last_wander_offset else None)
            chosen = next(
                ((s, a, o) for s, a, o in scored if o != reverse),
                scored[0],  # 他に選択肢がなければやむなく逆方向
            )
            best_score, best_action, offset = chosen
            self.trace['tactical'] += f' | move {offset} (score={best_score:.0f})'
            self.last_wander_offset = offset
            return best_action

        # 隣接に有望地なし → ブロック単位の品質で方角を選ぶ
        return self._move_by_block_quality(prompt, moves, prefer='quality')

    def _escape_from_cities(self, tiles, prompt, avail):
        """
        既存都市から遠ざかる方向に移動する。

        Settler が長期放浪して都市密集地に閉じ込められたとき、
        build_city が使えない状況から脱出するために使う。
        自都市タイルが視野にあればそこから遠ざかり、
        なければブロック品質で方向を選ぶ。
        """
        city_tiles = [t for t in tiles.values() if t.own_city and t.distance > 0]
        moves = self._available_moves(avail)
        if not moves:
            return None

        if not city_tiles:
            return self._move_by_block_quality(prompt, moves, prefer='quality')

        # 最寄り都市との距離が最大になる方向を選ぶ
        best, best_dist = None, -1
        for (dx, dy), action in moves.items():
            tile = tiles.get((dx, dy))
            if tile is not None and (tile.is_ocean or tile.own_city):
                continue
            min_city_dist = min(
                max(abs(ct.dx - dx), abs(ct.dy - dy))
                for ct in city_tiles
            )
            if min_city_dist > best_dist:
                best_dist, best = min_city_dist, action

        return best or self._move_by_block_quality(prompt, moves, prefer='quality')

    # ------------------------------------------------------------
    # ATTACK — 勝率で判断（安全弁つき）
    # ------------------------------------------------------------
    def _tactic_attack(self, prompt, avail, kind):
        tiles = parse_zoom_in(prompt)
        attack_actions = self._find_all_starting(avail, 'attack')

        if not attack_actions:
            # 攻撃できない → 敵の方へ移動する
            self.trace['feasibility'] = 'no attack action available'
            enemy = self._nearest_enemy_offset(tiles)
            if enemy:
                move = self._move_toward(tiles, avail, enemy)
                if move:
                    self.trace['tactical'] = f'approach enemy at {enemy}'
                    return move
            return self._tactic_explore(prompt, avail)

        self.trace['feasibility'] = f'{len(attack_actions)} attack action(s)'

        my_name = self.trace.get('unit_type', '')
        my_stats = UNIT_STATS.get(my_name, _DEFAULT_STATS)
        my_att, _, my_hp, my_fp, my_cost = my_stats

        # 隣接する敵ユニットを列挙して勝率を計算する
        best = None   # (desire, chance, action, enemy_name)
        worst_chance = 1.0
        adjacent_enemies = [
            t for t in tiles.values()
            if t.distance == 1 and t.has_enemy
        ]

        for tile in adjacent_enemies:
            enemy_name = tile.enemy_units[0] if tile.enemy_units else 'unknown'
            e_stats = UNIT_STATS.get(_norm(enemy_name), _DEFAULT_STATS)
            _, e_def, e_hp, e_fp, e_cost = e_stats
            if tile.enemy_city:
                e_def = int(e_def * 1.5)

            chance = win_chance(my_att, my_hp, my_fp, e_def, e_hp, e_fp)
            worst_chance = min(worst_chance, chance)
            desire = int(((e_cost + my_cost) * chance - my_cost) * SHIELD_WEIGHTING)

            action = self._attack_action_for(tile, attack_actions, avail)
            if action and (best is None or desire > best[0]):
                best = (desire, chance, action, enemy_name)

        # 敵の情報が読めなかった場合は指示どおり最初の攻撃を実行する
        if best is None:
            self.trace['tactical'] = 'enemy unreadable -> follow order'
            return attack_actions[0]

        desire, chance, action, enemy_name = best
        self.trace['tactical'] = (f'vs {enemy_name}: win={chance:.1%} '
                                  f'desire={desire}')

        # --- 安全弁: 勝率が極端に低い攻撃は拒否する ---
        if chance < SUICIDE_WIN_CHANCE:
            self._reject(f'attack rejected: vs {enemy_name} '
                         f'win={chance:.1%} < {SUICIDE_WIN_CHANCE:.0%}')
            alt = (self._find_exact(avail, 'fortify')
                   or self._find_exact(avail, 'sentry'))
            if alt:
                return alt
            return self._retreat(tiles, avail)

        return action

    def _attack_action_for(self, tile, attack_actions, avail):
        """
        対象タイルに対応する攻撃アクションを返す。
        'attack North' 形式なら方向で照合し、'attack 2' 形式なら候補から選ぶ。
        """
        # 方向名つきの表記を試す
        for name, vec in DIR_VECTORS.items():
            if vec != (tile.dx, tile.dy):
                continue
            for a in attack_actions:
                tail = _norm(a).replace('attack', '').strip().replace(' ', '')
                if tail == name:
                    return a
        # 番号形式などは判別できないので先頭を返す
        return attack_actions[0] if attack_actions else None

    # ------------------------------------------------------------
    # DEFEND — 都市へ集結して要塞化
    # ------------------------------------------------------------
    def _tactic_defend(self, prompt, avail):
        tiles = parse_zoom_in(prompt)
        here = tiles.get((0, 0))

        # 自都市の上にいる
        if here is not None and here.own_city:
            act = self._find_exact(avail, 'fortify') or self._find_exact(avail, 'sentry')
            if act:
                self.is_fortified_on_city = True
                self.trace['tactical'] = 'on own city -> fortify'
                return act
            # fortify/sentry がない = すでに要塞化済み
            # keep_activity は CivRealm で常に valid（そのターンを終了させる）
            for idle in ('keep activity', 'wait', 'sleep'):
                act2 = self._find_exact(avail, idle)
                if act2:
                    self.trace['tactical'] = 'on own city -> already fortified, keep_activity'
                    return act2
            # 万一 keep_activity もなければ None → fallback へ
            if self.is_fortified_on_city:
                self.trace['tactical'] = 'on own city -> already fortified, hold'
                return None
        else:
            # 都市の上にいない → fortify 状態をリセット
            self.is_fortified_on_city = False

        # 見える範囲に自都市があれば向かう
        own_city_tiles = [t for t in tiles.values() if t.own_city and t.distance > 0]
        if own_city_tiles:
            target = min(own_city_tiles, key=lambda t: t.distance)
            move = self._move_toward(tiles, avail, (target.dx, target.dy))
            if move:
                self.trace['tactical'] = f'move to own city at ({target.dx},{target.dy})'
                return move

        # 都市が見えない → その場で防御態勢
        act = self._find_exact(avail, 'fortify') or self._find_exact(avail, 'sentry')
        if act:
            self.trace['tactical'] = 'no city visible -> fortify in place'
            return act
        return None

    # ------------------------------------------------------------
    # IMPROVE — 地形に応じた最適改善 (autoworkers.c)
    # ------------------------------------------------------------
    def _tactic_improve(self, prompt, avail):
        tiles = parse_zoom_in(prompt)
        here = tiles.get((0, 0))
        terrain = here.terrain if here else None
        extras = here.extras if here else []

        # 地形ごとの優先順（Freeciv の autoworkers.c の考え方）
        if terrain in ('grassland', 'plains', 'desert'):
            order = ['irrigation', 'irrigate', 'build road', 'road', 'mine']
        elif terrain in ('hills', 'mountains'):
            order = ['mine', 'build mine', 'build road', 'road', 'irrigation']
        elif terrain in ('forest', 'jungle', 'swamp'):
            order = ['cultivate', 'plant', 'build road', 'road', 'mine']
        elif terrain == 'tundra':
            order = ['irrigation', 'irrigate', 'build road', 'road']
        else:
            order = ['irrigation', 'irrigate', 'mine', 'build road', 'road',
                     'cultivate', 'plant']

        # 既に道路があるなら道路を後回しにする
        if 'road' in extras:
            order = [o for o in order if 'road' not in o]

        for name in order:
            act = self._find_exact(avail, name)
            if act:
                self.trace['tactical'] = f'terrain={terrain} -> {name}'
                return act

        # このタイルでは作業できない → 自都市の周辺へ移動
        self.trace['tactical'] = f'terrain={terrain}: nothing to do -> relocate'
        own_city_tiles = [t for t in tiles.values() if t.own_city and t.distance > 0]
        if own_city_tiles:
            target = min(own_city_tiles, key=lambda t: t.distance)
            move = self._move_toward(tiles, avail, (target.dx, target.dy))
            if move:
                return move
        return self._move_by_block_quality(prompt, self._available_moves(avail),
                                           prefer='quality')

    # ------------------------------------------------------------
    # EXPLORE — 未探索方向へ
    # ------------------------------------------------------------
    def _tactic_explore(self, prompt, avail):
        act = self._find_exact(avail, 'explore')
        if act:
            self.trace['tactical'] = 'explore action available'
            return act

        moves = self._available_moves(avail)
        if not moves:
            return None
        return self._move_by_block_quality(prompt, moves, prefer='unexplored')

    # ------------------------------------------------------------
    # PRODUCE — advisor 指定の品目を生産
    # ------------------------------------------------------------
    def _tactic_produce(self, target, prompt, avail):
        produce_actions = self._all_produce_actions(avail)
        if not produce_actions:
            self.trace['feasibility'] = 'no production action'
            return None

        # 1) advisor が明示した品目
        if target and not target.startswith('__'):
            act = self._find_produce(avail, target)
            if act:
                self.trace['tactical'] = f'advisor item "{target}" -> {act}'
                return act
            self.trace['tactical'] = f'advisor item "{target}" unavailable'

        # 2) カテゴリ指定（防御 / 攻撃）
        if target == '__defender__':
            act = self._pick_from_priority(avail, DEFENDER_PRIORITY)
            if act:
                self.trace['tactical'] = f'defender category -> {act}'
                return act
        if target == '__attacker__':
            act = self._pick_from_priority(avail, ATTACKER_PRIORITY)
            if act:
                self.trace['tactical'] = f'attacker category -> {act}'
                return act

        # 3) 現在の生産を継続する（方針が読めないときに最も無難）
        keep = self._current_production_action(prompt, avail)
        if keep:
            prev = self.trace.get('tactical', '')
            suffix = f' | fallback: keep {keep}' if prev else f'keep current production -> {keep}'
            self.trace['tactical'] = prev + suffix if prev else suffix
            return keep

        # 4) 何も判断材料がない → 先頭の生産アクション
        self.trace['tactical'] = 'no guidance -> first production action'
        return produce_actions[0]

    def _pick_from_priority(self, avail, priority):
        for name in priority:
            act = self._find_produce(avail, name)
            if act:
                return act
        return None

    def _current_production_action(self, prompt, avail):
        """
        現在の生産物を読み取り、その継続アクションを返す。

        実際の出現形式は2つある:
          都市プロンプト (city_obs_action.txt):
              "This city is producing Workers."
          ユニットプロンプトの cities_summary (qwen_mastaba_agent.py:260):
              "- Roma at (10,12): pop=3, ..., producing=Workers"

        なお qwen_mastaba_agent.py:436 が
            available_actions += ['produce ' + producing]
        を追加するため、'produce X' 形式の継続アクションが実在する。
        """
        text = prompt or ''
        item = None

        m = re.search(r'this city is producing\s+([\w.\- ]+?)\s*\.', text, re.I)
        if m:
            item = m.group(1).strip()
        if not item:
            m = re.search(r'\bproducing\s*=\s*([\w.\- ]+?)\s*(?:,|\n|$)', text, re.I)
            if m:
                item = m.group(1).strip()

        if not item or _norm(item) in ('nothing', 'none', ''):
            return None
        return self._find_produce(avail, item)

    # ============================================================
    # 移動ヘルパー
    # ============================================================
    def _move_toward(self, tiles, avail, offset):
        """目標オフセットに最も近づく移動を返す。"""
        tx, ty = offset
        moves = self._available_moves(avail)
        if not moves:
            return None
        best, best_dist = None, max(abs(tx), abs(ty))
        for (dx, dy), action in moves.items():
            tile = tiles.get((dx, dy))
            if tile is not None and tile.is_ocean:
                continue
            dist = max(abs(tx - dx), abs(ty - dy))
            if dist < best_dist:
                best_dist, best = dist, action
        return best

    def _retreat(self, tiles, avail):
        """敵から最も離れる方向へ移動する。"""
        moves = self._available_moves(avail)
        if not moves:
            return None
        enemies = [(t.dx, t.dy) for t in tiles.values() if t.has_enemy]
        if not enemies:
            return random.choice(list(moves.values()))
        best, best_score = None, -1
        for (dx, dy), action in moves.items():
            tile = tiles.get((dx, dy))
            if tile is not None and tile.is_ocean:
                continue
            score = min(max(abs(ex - dx), abs(ey - dy)) for ex, ey in enemies)
            if score > best_score:
                best_score, best = score, action
        return best or random.choice(list(moves.values()))

    @staticmethod
    def _block_matches_direction(block_offset, move_delta):
        """
        ブロックが移動方向と同じ側にあるか判定する。

        以前は演算子の優先順位に依存した式だったため意図が読み取れなかった。
        明示的に「各軸の符号が矛盾しないこと」を判定する。
          北へ移動 (dx=0, dy=-1) のとき:
            block N  (0,-1) -> True     block NE (1,-1) -> False (x が余分)
            block S  (0, 1) -> False
        """
        bx, by = block_offset
        dx, dy = move_delta

        def axis_ok(b, d):
            if b == 0:
                return True          # その軸に偏りがないブロックは常に可
            if d == 0:
                return False         # 移動しない軸に偏るブロックは対象外
            return (b > 0) == (d > 0)

        return axis_ok(bx, dx) and axis_ok(by, dy)

    def _move_by_block_quality(self, prompt, moves, prefer='quality'):
        """
        zoomed-out observation のブロック情報で方角を選ぶ。
        prefer='unexplored' なら未探索の多い方角、'quality' なら地形の良い方角。
        """
        if not moves:
            return None
        blocks = parse_zoom_out(prompt)
        if not blocks:
            return random.choice(list(moves.values()))

        best, best_score = None, -1.0
        for delta, action in moves.items():
            score = 0.0
            for offset, block in blocks.items():
                if offset == (0, 0):
                    continue
                if not self._block_matches_direction(offset, delta):
                    continue
                score += (block.unexplored if prefer == 'unexplored'
                          else block.block_quality())
            if score > best_score:
                best_score, best = score, action

        if best is None:
            return random.choice(list(moves.values()))
        self.trace['tactical'] = self.trace.get('tactical', '') + \
            f' | block {prefer}={best_score:.0f}'
        return best

    # ============================================================
    # 安全弁
    # ============================================================
    def _reject(self, reason):
        """安全弁の発動を記録する。実験の評価指標として集計する。"""
        entry = f'[{self.ctrl_type} {self.actor_id}] {reason}'
        self.safety_rejections.append(entry)
        self.trace['safety'] = reason
        agent = self.agent
        if agent is not None:
            try:
                if not hasattr(agent, 'safety_rejection_count'):
                    agent.safety_rejection_count = 0
                    agent.safety_rejection_log = []
                agent.safety_rejection_count += 1
                agent.safety_rejection_log.append(entry)
            except Exception:
                pass

    # ============================================================
    # 第4層: フォールバック
    # ============================================================
    def _fallback(self, avail, prompt):
        """意図が実行できないときの安全な既定行動。"""
        if self.ctrl_type == 'city':
            keep = self._current_production_action(prompt, avail)
            if keep:
                return keep
            prod = self._all_produce_actions(avail)
            if prod:
                return prod[0]

        for safe in ('fortify', 'sentry', 'keep activity', 'explore'):
            act = self._find_exact(avail, safe)
            if act:
                return act

        # 要塞化済みユニットはランダム移動を避ける（都市から離散しない）
        if self.is_fortified_on_city:
            non_moves = [a for a in avail
                         if not _norm(a).startswith('move')
                         and not _norm(a).startswith('goto')
                         and not _norm(a).startswith('attack')]
            if non_moves:
                return non_moves[0]
            # 移動しか選択肢がなければ先頭を返す（ランダムより決定論的）
            moves = list(self._available_moves(avail).values())
            if moves:
                return moves[0]

        moves = list(self._available_moves(avail).values())
        if moves:
            return random.choice(moves)
        return random.choice(avail)

    # ============================================================
    # 敵位置
    # ============================================================
    @staticmethod
    def _nearest_enemy_offset(tiles):
        enemies = [t for t in tiles.values() if t.has_enemy]
        if not enemies:
            return None
        t = min(enemies, key=lambda x: x.distance)
        return (t.dx, t.dy)


# ============================================================
# エージェント側での集計用ヘルパー
# ============================================================
def aggregate_safety_rejections(agent):
    """
    全 worker の安全弁発動回数を集計する。
    main.py で retry_stats と並べて記録すると評価指標になる。
    """
    total = getattr(agent, 'safety_rejection_count', 0)
    log = getattr(agent, 'safety_rejection_log', [])
    by_kind = {}
    for entry in log:
        if 'attack rejected' in entry:
            by_kind['attack'] = by_kind.get('attack', 0) + 1
        elif 'build city rejected' in entry:
            by_kind['build_city'] = by_kind.get('build_city', 0) + 1
        else:
            by_kind['other'] = by_kind.get('other', 0) + 1
    return {'total': total, 'by_kind': by_kind, 'samples': log[-20:]}