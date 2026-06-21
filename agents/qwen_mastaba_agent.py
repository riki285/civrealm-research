import os
import time
import threading

from .mastaba_agent import MastabaAgent
from .workers import QwenMastabaWorker
from agents.utils import format_minimap_grid, format_upper_map_grid
from agents.redundants.improvement_consts import IMPR_TYPES

_KEY_BUILDINGS = {
    'Granary', 'Barracks', 'Barracks II', 'Barracks III',
    'Temple', 'Library', 'Marketplace', 'City Walls',
    'Harbor', 'Cathedral', 'Colosseum', 'University', 'Factory',
    'Aqueduct', 'Courthouse', 'Palace',
}


class QwenMastabaAgent(MastabaAgent):
    """
    MastabaAgent の Qwen ローカル推論版。
    Azure GPT の代わりに vLLM 経由の Qwen3.6-27B-int4-AutoRound を使用する。

    構造:
      - strategy_maker (advisor): ターン開始時に全体戦略を生成
      - workers (controller): エンティティごとに行動を決定（スレッド並列）
    """

    def initialize_workers(self):
        self.strategy_maker = QwenMastabaWorker(role="advisor", max_tokens=2048)
        self.workers = {}
        self._city_map_cache = {}       # {city_id: upper_map} — 行動不能ターンも含めて保持
        self._city_producing_cache = {}  # {city_id: str} — 最後に確認した生産物名
        self._cities_summary_cache = (None, None)  # (turn, str) — ターン単位キャッシュ

    def add_entity(self, entity_type, entity_id):
        if (entity_type, entity_id) in self.workers:
            return
        self.workers[(entity_type, entity_id)] = QwenMastabaWorker(
            role="controller",
            ctrl_type=entity_type,
            actor_id=entity_id,
        )

    def regenerate_conflict_actions(self, observations, info):
        """コンフリクトしたエンティティだけ再決定する。全エンティティの再実行を防ぐ。"""
        self.process_observations_and_info(observations, info)
        conflicted = list(self.conflict_action_list)
        self.conflict_action_list = []

        if not conflicted:
            return

        llm_info = self.info.get('llm_info', {})
        active = [
            (ctrl_type, actor_id, llm_info[ctrl_type][actor_id])
            for ctrl_type, actor_id, _ in conflicted
            if ctrl_type in llm_info and actor_id in llm_info[ctrl_type]
        ]

        per_unit_timeout = 120
        num_workers = max(1, len(active))
        for ctrl_type, actor_id, _ in active:
            worker = self.workers.get((ctrl_type, actor_id))
            if worker is not None and hasattr(worker, 'query_timeout'):
                worker.query_timeout = per_unit_timeout * num_workers

        threads = []
        for ctrl_type, actor_id, actor_dict in active:
            t = threading.Thread(
                target=self.make_single_decision,
                args=(ctrl_type, actor_id, actor_dict)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def is_action_valid(self, info, action):
        """都市の生産アクションは llm_info から消えた後も受け入れる。"""
        ctrl_type, actor_id, action_name = action
        try:
            return super().is_action_valid(info, action)
        except (KeyError, TypeError):
            # actor_id が llm_info に存在しない（他アクション処理後に消えた）
            # 都市生産アクションは決定時点で有効だったので受け入れる
            if ctrl_type == "city" and action_name.startswith("produce "):
                return action_name
            return False

    def _get_city_stats_str(self, actor_id) -> str:
        """自都市の生産統計を人間可読な文字列にして返す。"""
        try:
            raw = self.observations['city'][actor_id]

            def v(key):
                val = raw.get(key)
                if val is None:
                    return None
                try:
                    return int(val.flat[0]) if hasattr(val, 'flat') else int(val)
                except Exception:
                    return None

            size          = v('size')
            food_surplus  = v('surplus_food')
            granary_turns = v('granary_turns')
            shield_surplus = v('surplus_shield')
            shield_stock  = v('shield_stock')
            gold_surplus  = v('surplus_gold')
            science       = v('bulbs')

            # 食料の状況を言葉で補足
            if food_surplus is not None and food_surplus < 0:
                food_note = f"{food_surplus:+d}/turn (SHRINKING)"
            elif granary_turns is not None and granary_turns > 0:
                food_note = f"{food_surplus:+d}/turn (grows in {granary_turns} turns)"
            else:
                food_note = f"{food_surplus:+d}/turn" if food_surplus is not None else "?"

            parts = [
                f"population={size if size is not None else '?'}",
                f"food_surplus={food_note}",
                f"production={shield_surplus if shield_surplus is not None else '?'} shields/turn"
                + (f" (stock={shield_stock})" if shield_stock is not None else ""),
                f"gold={gold_surplus:+d}/turn" if gold_surplus is not None else "gold=?",
                f"science={science}/turn" if science is not None else "science=?",
            ]
            return "City stats: " + ", ".join(parts) + "."
        except Exception:
            return ""

    def _is_own_city(self, raw, city_id=None) -> bool:
        """obs['city'][id] が自国都市かどうかを判定する。

        llm_info['city'] に含まれていれば確実に自国都市（T002 起動直後でも信頼可能）。
        含まれていない場合は owner フィールドで判定（T003 以降は信頼可能）。
        """
        if city_id is not None and city_id in self.info.get('llm_info', {}).get('city', {}):
            return True
        try:
            owner = raw.get('owner')
            if owner is None:
                return False
            owner_val = int(owner.flat[0]) if hasattr(owner, 'flat') else int(owner)
            return owner_val == self.info['my_player_id']
        except Exception:
            return False

    def _get_city_name_at(self, actor_x: int, actor_y: int) -> dict:
        """全自都市の {(dx,dy): city_name} を返す。ミニマップ・上位マップ両方で使用。"""
        result = {}
        llm_cities = self.info.get('llm_info', {}).get('city', {})
        for city_id, raw in self.observations.get('city', {}).items():
            if not self._is_own_city(raw, city_id):
                continue
            try:
                cx = raw.get('x'); cy = raw.get('y')
                if cx is None or cy is None:
                    continue
                cx = int(cx.flat[0]) if hasattr(cx, 'flat') else int(cx)
                cy = int(cy.flat[0]) if hasattr(cy, 'flat') else int(cy)
            except Exception:
                continue
            name = llm_cities.get(city_id, {}).get('name', f'city {city_id}')
            result[(cx - actor_x, cy - actor_y)] = name
        return result

    def _get_city_buildings_str(self, actor_id) -> str:
        """建設済み建造物のリストを文字列で返す。"""
        try:
            improvements = self.observations['city'][actor_id].get('improvements')
            if improvements is None:
                return ""
            built = [
                IMPR_TYPES[i]
                for i in range(min(len(IMPR_TYPES), len(improvements)))
                if improvements[i] == 1
            ]
            if not built:
                return "Buildings already built: none."
            return "Buildings already built: " + ", ".join(built) + "."
        except Exception:
            return ""

    def _get_all_cities_summary(self) -> str:
        """全自都市の概要を1行ずつまとめた文字列を返す。同一ターン内は計算結果をキャッシュする。"""
        current_turn = self.info.get('turn') if self.info else None
        cached_turn, cached_str = self._cities_summary_cache
        if cached_turn is not None and cached_turn == current_turn:
            return cached_str
        try:
            lines = []
            city_obs = self.observations.get('city', {})
            llm_cities = self.info.get('llm_info', {}).get('city', {})

            for city_id, raw in city_obs.items():
                if not self._is_own_city(raw, city_id):
                    continue

                city_info = llm_cities.get(city_id, {})
                name = city_info.get('name', f'City {city_id}')
                _p = (city_info.get('observations', {}).get('producing') if city_info else None)
                if _p is not None:
                    self._city_producing_cache[city_id] = _p
                producing = self._city_producing_cache.get(city_id)

                def v(key, r=raw):
                    val = r.get(key)
                    if val is None:
                        return None
                    try:
                        return int(val.flat[0]) if hasattr(val, 'flat') else int(val)
                    except Exception:
                        return None

                size = v('size')
                food_surplus = v('surplus_food')
                granary_turns = v('granary_turns')
                shield_surplus = v('surplus_shield')
                gold_surplus = v('surplus_gold')
                science = v('bulbs')

                if food_surplus is not None and food_surplus < 0:
                    food_note = f"{food_surplus:+d}/turn(SHRINKING)"
                elif granary_turns is not None and granary_turns > 0:
                    food_note = f"{food_surplus:+d}/turn(grows in {granary_turns}t)"
                else:
                    food_note = f"{food_surplus:+d}/turn" if food_surplus is not None else "?"

                # 主要建造物（済）
                impr_arr = raw.get('improvements')
                if impr_arr is not None:
                    built_key = [
                        IMPR_TYPES[i]
                        for i in range(min(len(IMPR_TYPES), len(impr_arr)))
                        if impr_arr[i] == 1 and IMPR_TYPES[i] in _KEY_BUILDINGS
                    ]
                    buildings_note = f", buildings=[{', '.join(built_key)}]" if built_key else ", buildings=[]"
                else:
                    buildings_note = ""

                try:
                    cx = raw.get('x'); cy = raw.get('y')
                    cx = int(cx.flat[0]) if hasattr(cx, 'flat') else int(cx)
                    cy = int(cy.flat[0]) if hasattr(cy, 'flat') else int(cy)
                    pos_str = f" at ({cx},{cy})"
                except Exception:
                    pos_str = ""

                line = (
                    f"- {name}{pos_str}: pop={size if size is not None else '?'}"
                    f", food={food_note}"
                    f", shields={shield_surplus if shield_surplus is not None else '?'}/turn"
                    f", gold={f'{gold_surplus:+d}/turn' if gold_surplus is not None else '?'}"
                    f", science={science if science is not None else '?'}/turn"
                    + (f", producing={producing}" if producing is not None else "")
                    + f"{buildings_note}"
                )
                lines.append(line)

            if not lines:
                self._cities_summary_cache = (current_turn, "")
                return ""
            result = "Our cities:\n" + "\n".join(lines)
            self._cities_summary_cache = (current_turn, result)
            return result
        except Exception:
            return ""

    def get_advisor_input_prompt(self, obs, info):
        """アドバイザー用プロンプト: 統計情報 + 全都市zoom-out + 都市詳細。"""
        # --- 統計情報（基底クラスと同じロジック）---
        munit_num_self = wunit_num_self = unit_num_enemy = 0
        city_num_self = city_size_self = city_num_other = city_num_enemy = 0
        units = {}

        for key, val in obs['unit'].items():
            if val['owner'] == info['my_player_id']:
                unit_name = val['type_rule_name']
                units[unit_name] = units.get(unit_name, 0) + 1
                if val['type_attack_strength'] == 0:
                    wunit_num_self += 1
                else:
                    munit_num_self += 1
            dipl_entry = obs['dipl'].get(val['owner'])
            if dipl_entry and dipl_entry.get('diplomatic_state') == 1:
                unit_num_enemy += 1

        my_id = info['my_player_id']
        for key, val in obs['city'].items():
            try:
                owner = int(val['owner'].flat[0]) if hasattr(val['owner'], 'flat') else int(val['owner'])
            except Exception:
                continue
            if owner == my_id:
                city_num_self += 1
                try:
                    city_size_self += int(val['size'].flat[0]) if hasattr(val['size'], 'flat') else int(val['size'])
                except Exception:
                    pass
                continue
            try:
                vx = int(val['x'].flat[0]) if hasattr(val['x'], 'flat') else int(val['x'])
                vy = int(val['y'].flat[0]) if hasattr(val['y'], 'flat') else int(val['y'])
            except Exception:
                continue
            if obs['map']['status'][vx, vy] <= 1:
                continue
            dipl_entry = obs['dipl'].get(owner)
            if dipl_entry and dipl_entry.get('diplomatic_state') == 1:
                city_num_enemy += 1
            else:
                city_num_other += 1

        if unit_num_enemy > munit_num_self / 5 and city_num_enemy < unit_num_enemy:
            war_state = "We are under attack."
        elif city_num_enemy >= 3 and unit_num_enemy < munit_num_self:
            war_state = "We are attacking other players."
        elif unit_num_enemy == 0 and city_num_enemy < 4:
            war_state = "We are in peace."
        else:
            war_state = "We are roughly safe."

        unit_spec = (
            f"We have {wunit_num_self + munit_num_self} units: "
            + ", ".join(f"{n} {u}" for u, n in units.items())
        )
        stats_text = " ".join([
            unit_spec, f"and we can see {unit_num_enemy} enemy units.",
            f"We have {city_num_self} cities of total size {city_size_self}.",
            f"We can see {city_num_enemy} enemy cities, and {city_num_other} other cities.",
            war_state,
        ])

        # --- 全自都市の zoom-out マップ ---
        llm_cities = info.get('llm_info', {}).get('city', {})
        own_cities = []
        for city_id, raw in obs.get('city', {}).items():
            if not self._is_own_city(raw, city_id):
                continue
            try:
                cx = raw.get('x'); cy = raw.get('y')
                cx = int(cx.flat[0]) if hasattr(cx, 'flat') else int(cx)
                cy = int(cy.flat[0]) if hasattr(cy, 'flat') else int(cy)
            except Exception:
                continue
            city_info = llm_cities.get(city_id, {})
            name = city_info.get('name', f'city {city_id}')
            upper_map = (city_info.get('observations', {}).get('upper_map') if city_info else None) \
                or self._city_map_cache.get(city_id)
            # キャピタル判定（Palace保有）
            impr = raw.get('improvements')
            is_capital = False
            if impr is not None:
                try:
                    palace_idx = IMPR_TYPES.index('Palace')
                    is_capital = bool(impr[palace_idx] == 1)
                except (ValueError, IndexError):
                    pass
            own_cities.append({
                'id': city_id, 'name': name, 'x': cx, 'y': cy,
                'is_capital': is_capital, 'upper_map': upper_map,
            })

        # キャピタルを先頭に、残りはID順
        own_cities.sort(key=lambda c: (not c['is_capital'], c['id']))
        capital = own_cities[0] if own_cities else None

        map_sections = []
        for c in own_cities:
            if c['upper_map'] is None:
                continue
            city_name_at = self._get_city_name_at(c['x'], c['y'])
            zoom_out = format_upper_map_grid(c['upper_map'], city_name_at)

            label_parts = [c['name']]
            if c['is_capital']:
                label_parts.append('capital')
                label_parts.append('at (0, 0)')
            elif capital:
                dx = c['x'] - capital['x']
                dy = c['y'] - capital['y']
                map_status = obs.get('map', {}).get('status')
                if map_status is not None:
                    map_w, map_h = map_status.shape[0], map_status.shape[1]
                    if abs(dx) > map_w / 2:
                        dx = dx - map_w if dx > 0 else dx + map_w
                    if abs(dy) > map_h / 2:
                        dy = dy - map_h if dy > 0 else dy + map_h
                label_parts.append(f'at ({dx:+d}, {dy:+d})')
            label = ', '.join(label_parts)
            map_sections.append(f"--- {label} ---\n{zoom_out}")

        cities_summary = self._get_all_cities_summary()

        parts = [stats_text]
        if map_sections:
            parts.append('\n'.join(map_sections))
        if cities_summary:
            parts.append(cities_summary)
        return '\n'.join(parts)

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict, available_actions, actor_id=None):
        # アクター位置を取得し、ミニマップ内の自都市に名前を付ける
        city_name_at = {}
        ax = ay = None
        if actor_id is not None:
            try:
                raw = self.observations.get(ctrl_type, {}).get(actor_id, {})
                ax = raw.get('x'); ay = raw.get('y')
                if ax is not None and ay is not None:
                    ax = int(ax.flat[0]) if hasattr(ax, 'flat') else int(ax)
                    ay = int(ay.flat[0]) if hasattr(ay, 'flat') else int(ay)
                    city_name_at = self._get_city_name_at(ax, ay)
            except Exception:
                ax = ay = None

        zoom_in_obs = format_minimap_grid(actor_dict['observations']['minimap'], city_name_at)
        zoom_out_obs = format_upper_map_grid(actor_dict['observations']['upper_map'], city_name_at)

        # 前ターンの行動をアドバイスの先頭に追記（Fix2: 実際に実行された行動を参照）
        prev_note = ''
        if actor_id is not None:
            taken = self.last_taken_actions.get((ctrl_type, actor_id))
            if taken:
                prev_note = f'Your previous action was: {taken[0]}. '
        advise = prev_note + (self.general_advise or '')

        if ctrl_type == "city":
            producing = actor_dict['observations'].get('producing', "NOTHING")
            if producing and producing != "NOTHING":
                available_actions = list(available_actions) + ['produce ' + producing]
            else:
                available_actions = list(available_actions)
            city_stats = self._get_city_stats_str(actor_id) if actor_id is not None else ""
            city_buildings = self._get_city_buildings_str(actor_id) if actor_id is not None else ""
            # upper_mapとproducingをキャッシュ（行動できないターンでもadvisorが参照できるように）
            if actor_id is not None:
                upper_map = actor_dict['observations'].get('upper_map')
                if upper_map is not None:
                    self._city_map_cache[actor_id] = upper_map
                if producing and producing != "NOTHING":
                    self._city_producing_cache[actor_id] = producing
            prompt = self.strategy_maker.prompt_handler.city_obs_action(
                ctrl_type=ctrl_type,
                actor_name=actor_name,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                producing=producing,
                city_stats=city_stats,
                city_buildings=city_buildings,
                available_actions=available_actions,
                general_advise=advise,
            )
        else:
            cities_summary = self._get_all_cities_summary()
            if ax is not None and ay is not None:
                cities_summary += f"\nYour position: ({ax},{ay})."
            prompt = self.strategy_maker.prompt_handler.unit_obs_action(
                ctrl_type=ctrl_type,
                actor_name=actor_name,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                cities_summary=cities_summary,
                available_actions=available_actions,
                general_advise=advise,
            )
        return prompt
