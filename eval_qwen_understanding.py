"""
Qwen の Freeciv アクション理解テスト。

使い方:
  python eval_qwen_understanding.py              # テスト実行 + プロンプト補足案を出力
  python eval_qwen_understanding.py --with-prompt # instruction_prompt.txt を読み込んだ状態でテスト
  python eval_qwen_understanding.py --dry-run     # Qwen に送らず質問集だけ表示

vLLM サーバーが localhost:8001 で動いていること。
"""

import argparse
import json
import os
import re
import requests

API_URL = "http://localhost:8001/v1/chat/completions"
MODEL   = "Lorbus/Qwen3.6-27B-int4-AutoRound"

INSTRUCTION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_collections/civ_prompts_qwen/instruction_prompt.txt"
)

# ---------------------------------------------------------------------------
# 質問集
# 各エントリ:
#   question    : Qwen に送るテキスト
#   correct     : 正解の要点（キーワードリスト。すべてが回答に含まれれば正解）
#   wrong_hints : 誤答パターンのキーワードリスト
#   topic       : 何を測るか
#   correct_answer : 正解の説明（Qwen に追記案を生成させる際に使用）
# ---------------------------------------------------------------------------
def make_question(stem, choices, correct_index, correct_answer):
    """choices は4つのリスト。correct_index は 1〜4。"""
    opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    question = f"{stem}\n{opts}\nAnswer with just the number 1, 2, 3, or 4."
    return {
        "question": question,
        "choices": choices,
        "correct": str(correct_index),
        "correct_label": choices[correct_index - 1],
        "correct_answer": correct_answer,
    }


QUESTIONS = [
    # Q1: 食料危機への Worker 対応（irrigate の必要性）
    {
        "topic": "starving city 隣の Worker - 最適な改良アクション",
        **make_question(
            "City 121 has food=-2/turn and will lose population in 3 turns. "
            "A Worker is on an adjacent Grassland tile (food=2, no improvements). "
            "Available: ['plant', 'irrigate', 'mine', 'build road']. "
            "Which action directly increases this tile's food output?",
            ["plant", "irrigate", "mine", "build road"],
            correct_index=2,
            correct_answer=(
                "'irrigate' adds +1 food to Grassland (food: 2→3), relieving the starvation. "
                "'plant' converts Grassland to Forest (food: 2→1), making it worse. "
                "Use 'irrigate' whenever a city is food-deficient."
            ),
        ),
    },
    # Q2: Granary の食料保存効果
    {
        "topic": "Granary は人口増加後の食料バーをどう変えるか",
        **make_question(
            "A city needs 20 food to grow from population=2 to population=3. "
            "WITHOUT Granary: food bar resets to 0 after growing. "
            "WITH Granary: food bar resets to __ after growing.",
            ["0 (same as without Granary)",
             "10 (Granary stores half the threshold)",
             "20 (bar stays full)",
             "5 (one quarter)"],
            correct_index=2,
            correct_answer=(
                "Granary stores half the food bar on population growth (resets to 10 instead of 0). "
                "This halves the turns needed for each subsequent population increase. "
                "Granary is the highest-priority building for food-limited cities after reaching 6 cities."
            ),
        ),
    },
    # Q3: Explorer の役割（戦闘不可ユニット）
    {
        "topic": "Explorer が敵ユニットと隣接した時の行動",
        **make_question(
            "An Explorer unit is used for scouting and cannot survive combat. "
            "An enemy Warrior appears in an adjacent tile. "
            "Available: ['attack', 'move West', 'move South', 'fortify']. "
            "What should the Explorer do?",
            ["attack the enemy Warrior",
             "fortify in place to block enemy movement",
             "move West or South to escape immediately",
             "wait and hope the enemy ignores it"],
            correct_index=3,
            correct_answer=(
                "Explorers cannot fight and die to any attack. "
                "The only correct action is to move away immediately. "
                "Never attack or stay adjacent to enemies with an Explorer."
            ),
        ),
    },
    # Q4: Settlers と Workers の役割分担
    {
        "topic": "Settlers にしかできないこと — 最適な役割分担",
        **make_question(
            "You have 1 Settler and 1 Worker. You need to: "
            "(A) Irrigate Grassland near City A. "
            "(B) Found a new city on a distant Grassland tile. "
            "How should you assign these two units?",
            ["Settler irrigates City A; Worker founds the new city",
             "Settler founds the new city; Worker irrigates City A",
             "Both units irrigate first, then found the city together",
             "Worker founds the city; Settler also irrigates afterward"],
            correct_index=2,
            correct_answer=(
                "Only Settlers can found new cities — Workers cannot. "
                "Workers specialize in tile improvements (irrigate, mine, road). "
                "Always use Settlers for city founding and Workers for improvements."
            ),
        ),
    },
    # Q5: 攻撃中の建物生産 vs Warriors 切り替え
    {
        "topic": "敵が 5 ターン以内に接近。Granary 生産中の都市はどうすべきか",
        **make_question(
            "City B (5 shields/turn) is producing Granary (cost=60 shields, 8 turns remaining). "
            "An enemy unit is 5 turns away. Warriors cost 10 shields (2 turns to produce). "
            "The advisor says: 'Switch to Warriors immediately.' What should City B do?",
            ["Continue Granary (60% done, don't waste invested shields)",
             "Switch to Warriors immediately (ready in 2 turns, before enemy arrives)",
             "Switch to Phalanx (stronger than Warriors for defense)",
             "Switch to Settlers to evacuate the population"],
            correct_index=2,
            correct_answer=(
                "Switch to Warriors immediately — they take 2 turns and are ready before the enemy. "
                "Granary has no combat value; invested shields are a sunk cost. "
                "When under attack, producing a military unit always overrides economic buildings."
            ),
        ),
    },
    # Q6: Warriors — 敵が 3 タイル先にいる場合の最善行動
    {
        "topic": "都市内 Warriors。敵が 3 タイル先に見える — 出撃すべきか",
        **make_question(
            "A Warriors unit is fortified inside a city. "
            "An enemy unit is visible 3 tiles away and moving toward the city. "
            "Available: ['move North', 'move East', 'move West', 'fortify']. "
            "What should the Warriors do?",
            ["Move North to intercept the enemy in open field",
             "Move East to flank the enemy from the side",
             "Move West to draw the enemy away from the city",
             "Stay fortified inside the city (defense bonus + city walls)"],
            correct_index=4,
            correct_answer=(
                "Staying fortified inside the city gives Warriors a 2x defense bonus plus city tile protection. "
                "Moving out into open field removes these bonuses and risks the city being taken undefended. "
                "Always wait for the enemy to attack into the city — defenders have the advantage."
            ),
        ),
    },
    # Q7: road の軍事的メリット
    {
        "topic": "road を建設した主な軍事的メリットは何か",
        **make_question(
            "A Worker builds a road connecting City A and City B (5 tiles apart). "
            "From a military perspective, what is the most important benefit of this road?",
            ["Units gain +1 attack when moving along roads",
             "Units can move faster along roads, enabling rapid reinforcement between cities",
             "Roads add +2 food to all adjacent city tiles",
             "Enemy units cannot enter road tiles without paying extra movement cost"],
            correct_index=2,
            correct_answer=(
                "Roads allow units to move faster — multiple tiles per turn along roads. "
                "This is critical for rapidly reinforcing threatened cities. "
                "Roads do NOT add food, buff attacks, or block enemy movement."
            ),
        ),
    },
    # Q8: population=1 の都市が starvation した結果
    {
        "topic": "population=1 の都市が食料不足になるとどうなるか",
        **make_question(
            "City C has population=1 and food=-1/turn. "
            "No Worker can reach the city in time to fix the food shortage. "
            "What happens when the food bar reaches 0?",
            ["The city is permanently destroyed (population cannot go below 1)",
             "The city survives at population=1 but produces nothing for 5 turns",
             "The city automatically builds a Granary to fix the food",
             "Population stays at 1 and gold decreases instead of food"],
            correct_index=1,
            correct_answer=(
                "A city at population=1 that starves is permanently destroyed — all buildings and progress are lost. "
                "This is the worst outcome in the game. "
                "Preventing starvation in small cities is the highest priority Worker task."
            ),
        ),
    },
    # Q9: Barracks の効果
    {
        "topic": "Barracks を建設すると生産ユニットにどんな効果があるか",
        **make_question(
            "City D builds a Barracks. From now on, all military units produced in City D will:",
            ["Cost 0 shields (free production)",
             "Start as Veterans with higher combat effectiveness",
             "Move 2 tiles per turn instead of 1",
             "Automatically fortify upon being produced"],
            correct_index=2,
            correct_answer=(
                "Barracks make all units produced in the city start as Veterans. "
                "Veterans have higher attack/defense power and more hit points than non-veterans. "
                "Barracks is the highest-priority military building — build it before producing large numbers of units."
            ),
        ),
    },
    # Q10: 人口増加の経済的メリット
    {
        "topic": "population が増えると都市の資源産出はどう変わるか",
        **make_question(
            "City E grows from population=2 to population=3. "
            "Each citizen works one adjacent tile. "
            "What is the direct economic benefit of gaining one more citizen?",
            ["The city gains 1 additional Warrior automatically",
             "The city can work one more adjacent tile, adding its food/shields/trade",
             "All existing tile outputs double",
             "The city generates +5 gold per turn from population tax"],
            correct_index=2,
            correct_answer=(
                "Each new citizen works one additional adjacent tile, adding its resources to the city total. "
                "More population → more tiles worked → more food, shields, and trade. "
                "Population growth (via food surplus and Granary) is the foundation of all long-term progress."
            ),
        ),
    },
]

# ---------------------------------------------------------------------------

def query_qwen(messages):
    try:
        resp = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 300,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")
    return None


def extract_answer(raw, choices=None):
    """JSON action フィールド / 生テキストから数字（1〜4）を取り出す。
    数字がない場合は choices テキストとの照合で逆引きする。"""
    action_text = None

    # JSON action フィールドを試みる
    try:
        j = json.loads(raw)
        val = j.get("command", {}).get("input", {}).get("action", "").strip()
        if val:
            m = re.search(r"[1-4]", val)
            if m:
                return m.group()
            action_text = val  # 数字なし → choices 照合へ
    except Exception:
        pass

    # 生テキストから独立した数字（1〜4）を取り出す
    m = re.search(r"\b([1-4])\b", raw)
    if m:
        return m.group(1)

    # 選択肢テキストとの照合で逆引き（JSON action 値 or 生テキストで試みる）
    if choices:
        for search in ([action_text] if action_text else []) + [raw.strip()]:
            search_lower = search.lower()
            for i, choice in enumerate(choices, 1):
                if choice.lower() in search_lower or search_lower in choice.lower():
                    return str(i)

    return (action_text or raw.strip())[:20]


def evaluate(answer, correct, choices=None):
    got = extract_answer(answer, choices)
    passed = got == correct
    return passed, got


def load_instruction_prompt():
    with open(INSTRUCTION_PROMPT_PATH) as f:
        return f.read()


def generate_fix(question, wrong_answer, correct_answer):
    """Qwen 自身に追記案を生成させる。"""
    prompt = (
        "You are helping improve an AI agent's instruction prompt for the FreeCiv game.\n\n"
        f"The agent was asked:\n{question}\n\n"
        f"The agent answered incorrectly:\n{wrong_answer}\n\n"
        f"The correct answer is:\n{correct_answer}\n\n"
        "Write ONE concise sentence (max 30 words) to add to the instruction prompt "
        "that would prevent this mistake. "
        "Start directly with the rule — no preamble, no bullet point."
    )
    resp = requests.post(
        API_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return "(追記案の生成に失敗しました)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-prompt", action="store_true",
                        help="instruction_prompt.txt をシステムプロンプトとして付加する")
    parser.add_argument("--dry-run", action="store_true",
                        help="Qwen に送らず質問だけ表示する")
    args = parser.parse_args()

    system_prompt = None
    if args.with_prompt:
        system_prompt = load_instruction_prompt()
        print("=== instruction_prompt.txt をシステムプロンプトとして使用 ===\n")
    else:
        print("=== システムプロンプトなし（Qwen の素の理解をテスト）===\n")

    failed_topics = []

    SEP = "-" * 60

    for i, q in enumerate(QUESTIONS, 1):
        print(SEP)
        print(f"[Q{i}/{len(QUESTIONS)}] {q['topic']}")
        print(f"  質問: {q['question']}")
        if args.dry_run:
            print()
            continue

        messages = []
        if system_prompt:
            # ゲーム知識は維持しつつ、評価モードでは数字のみを要求する
            eval_system = (
                system_prompt.split("Rules:")[0].strip()
                + "\n\nFor this knowledge test, respond with ONLY a single digit: "
                "1, 2, 3, or 4. No JSON, no explanations, just the digit."
            )
            messages.append({"role": "system", "content": eval_system})
        else:
            messages.append({"role": "system", "content":
                "Respond with ONLY a single digit: 1, 2, 3, or 4. Nothing else."})
        messages.append({"role": "user", "content": q["question"]})

        answer = query_qwen(messages)
        if answer is None:
            print("  → [ERROR] 応答なし")
            print()
            continue

        # JSON の場合は reasoning だけ抜き出して表示
        import json as _json
        display_answer = answer
        try:
            j = _json.loads(answer)
            th = j.get("thoughts", {})
            display_answer = (
                f'thought   : {th.get("thought", "")}\n'
                f'  reasoning : {th.get("reasoning", "")}\n'
                f'  plan      : {th.get("plan", "")}\n'
                f'  action    : {j.get("command", {}).get("input", {}).get("action", "(no action field)")}'
            )
        except Exception:
            display_answer = answer[:300]

        passed, got = evaluate(answer, q["correct"], q.get("choices"))
        status = "PASS ✅" if passed else "FAIL ❌"
        correct_label = q.get("correct_label", "")
        print(f"  正解: {q['correct']}（{correct_label}）  /  Qwen: {got}  → [{status}]")
        if not passed:
            print(f"  📝 追記案を生成中...")
            fix = generate_fix(q["question"], answer, q["correct_answer"])
            print(f"  ┌─ instruction_prompt への追記案 ─────────────────")
            print(f"  │ {fix}")
            print(f"  └──────────────────────────────────────────────")
            failed_topics.append((q, fix))
        print()

    if not args.dry_run:
        print("=" * 60)
        if failed_topics:
            print(f"結果: {len(QUESTIONS) - len(failed_topics)}/{len(QUESTIONS)} PASS  "
                  f"({len(failed_topics)} 問失敗)")
        else:
            print("全問正解！instruction_prompt の追記は不要です。")


if __name__ == "__main__":
    main()
