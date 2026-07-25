import random


class RuleBasedWorker:
    """
    LLM を使わないルールベースの controller ワーカー。
    QwenMastabaAgent._create_controller_worker() から生成される。

    choose_action(input_prompt, avail_action_list) を実装し、
    BaseLangAgent.make_single_decision() と互換のインターフェースを持つ。
    """

    # ユニット種別ごとのアクション優先順位
    _PRIORITY = {
        "Settlers": [
            "build city",
        ],
        "Workers": [
            "irrigation", "build irrigation",
            "build road",
            "mine", "build mine",
            "transform terrain",
            "build fortress",
        ],
        "Warriors": ["fortify", "move North", "move South", "move East", "move West",
                     "move NorthEast", "move NorthWest", "move SouthEast", "move SouthWest"],
        "Archers":  ["fortify"],
        "Phalanx":  ["fortify"],
        "Musketeers": ["fortify"],
        "Explorer": ["move North", "move South", "move East", "move West",
                     "move NorthEast", "move NorthWest", "move SouthEast", "move SouthWest"],
    }

    # どの種別にも共通の移動アクション（フォールバック用）
    _MOVE_ACTIONS = [
        "move North", "move NorthEast", "move East", "move SouthEast",
        "move South", "move SouthWest", "move West", "move NorthWest",
    ]

    def __init__(self, ctrl_type: str = "unit", actor_id: int = -1):
        self.ctrl_type = ctrl_type
        self.actor_id = actor_id
        self.last_retry_count = 0  # BaseLangAgent の retry_stats 集計に必要

    def choose_action(self, input_prompt: str, avail_action_list: list) -> str:
        unit_type = self._parse_unit_type(input_prompt)
        action = self._select_by_priority(unit_type, avail_action_list)
        if action is None:
            action = random.choice(avail_action_list)
        return action

    def _parse_unit_type(self, input_prompt: str) -> str:
        """input_prompt の "You are controlling unit: <Type> <id>." から種別を抽出する。"""
        import re
        m = re.search(r'controlling unit:\s+(\w+)', input_prompt)
        if m:
            return m.group(1)
        # 都市の場合
        if "controlling city" in input_prompt or self.ctrl_type == "city":
            return "city"
        return "unknown"

    def _select_by_priority(self, unit_type: str, avail: list):
        """優先リストの順にavail_action_listを検索し、最初に見つかったものを返す。"""
        avail_set = set(avail)

        priority_list = self._PRIORITY.get(unit_type, [])
        for action in priority_list:
            if action in avail_set:
                return action

        # 都市は produce 系アクションを優先
        if unit_type == "city":
            for action in avail:
                if action.startswith("produce "):
                    return action

        # 移動アクションをフォールバックとして返す
        for action in self._MOVE_ACTIONS:
            if action in avail_set:
                return action

        return None
