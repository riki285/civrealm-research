import json
import requests

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler
from agents.civ_autogpt.utils import num_tokens_from_messages, TOKEN_LIMIT_TABLE
from .base_worker import BaseWorker

# GPT版の ConversationSummaryBufferMemory(max_token_limit=500) に相当する文字数しきい値
# 500 tokens ≈ 2000 chars
_MEMORY_CHAR_LIMIT = 2000


class QwenWorker(BaseWorker):
    """
    Worker for local Qwen (vLLM) inference.
    Follows the same structure as AzureGPTWorker.
    """

    def __init__(self,
                 model: str = "Lorbus/Qwen3.6-27B-int4-AutoRound",
                 api_url: str = "http://localhost:8001/v1/chat/completions",
                 prompt_prefix: str = "civ_prompts_qwen",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 **kwargs):
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_prefix = prompt_prefix
        self.query_timeout = 120  # make_decisions() で num_workers × 120 に上書きされる
        super().__init__(model, **kwargs)

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(prompt_prefix=self.prompt_prefix)
        self.add_user_message_to_dialogue(self.prompt_handler.instruction_prompt())
        self.add_user_message_to_dialogue(self.prompt_handler.task_prompt())

    def init_llm(self):
        # GPT版の ConversationSummaryBufferMemory 相当。
        # 毎ターン _update_memory() でインクリメンタルに更新される文字列。
        self.memory = ""

    def init_index(self):
        self.index = None

    # ------------------------------------------------------------------
    # コマンド登録
    # ------------------------------------------------------------------
    def register_all_commands(self):
        self.register_command('finalDecision', self.handle_command_final_decision)
        self.register_command('suggestion',    self.handle_command_suggestion)

    def handle_command_suggestion(self, command_input, obs_input_prompt, current_avail_actions):
        return command_input['suggestion'], ''

    def handle_command_final_decision(self, command_input, obs_input_prompt, current_avail_actions):
        exec_action = command_input['action']
        lower_avail = [x.lower() for x in current_avail_actions]

        if exec_action.lower() not in lower_avail:
            fc_logger.error(f'Invalid action "{exec_action}", retrying...')
            return None, self.prompt_handler.insist_avail_action()

        self.taken_actions_list.append(exec_action)
        return exec_action, ''

    # ------------------------------------------------------------------
    # vLLM呼び出し
    # ------------------------------------------------------------------
    def query_llm(self):
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model":       self.model,
                    "messages":    self.dialogue,
                    "temperature": self.temperature,
                    "max_tokens":  self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=self.query_timeout
            )

            print("Status:", response.status_code)

            if response.status_code != 200:
                print("Error response:", response.text)
                return None

            return response.json()

        except Exception as e:
            print("Exception:", e)
            return None

    # ------------------------------------------------------------------
    # 会話履歴管理（GPT版 ConversationSummaryBufferMemory 相当）
    # ------------------------------------------------------------------
    def _update_memory(self, user_prompt: str, assistant_content: str):
        """今ターンのやり取りをmemoryに追記し、長くなりすぎたら再要約する。
        GPT版の memory.save_context() 相当。毎ターン generate_command() から呼ばれる。"""
        new_entry = (
            f"USER: {user_prompt[:800]}\n"
            f"ASSISTANT: {assistant_content[:500]}"
        )
        updated = (self.memory + "\n\n" + new_entry).strip() if self.memory else new_entry

        if len(updated) <= _MEMORY_CHAR_LIMIT:
            self.memory = updated
            return

        # しきい値超過 → Qwenで再要約（GPT版が LLM で summary を更新するのと同等）
        summary_prompt = (
            "The following is a Freeciv game conversation history for one unit/city. "
            "Summarize it concisely (under 150 words), focusing on: "
            "what actions were taken, what was observed, and any strategic patterns.\n\n"
            f"{updated}"
        )
        try:
            resp = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 250,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=self.query_timeout
            )
            if resp.status_code == 200:
                self.memory = resp.json()['choices'][0]['message']['content']
                return
        except Exception as e:
            fc_logger.error(f'Memory update summarization failed: {e}')
        # フォールバック: 後半を残す
        self.memory = updated[-_MEMORY_CHAR_LIMIT:]

    def restrict_dialogue(self):
        """トークン上限超過時に memory 要約を dialogue に注入する。
        GPT版 base_worker.restrict_dialogue() と同等。"""
        model_ctx = 16384  # Lorbus/Qwen3.6-27B-int4-AutoRound の最大コンテキスト長
        limit = min(TOKEN_LIMIT_TABLE.get(self.model, 14000), model_ctx - self.max_tokens)
        while num_tokens_from_messages(self.dialogue, self.model) >= limit:
            if not self.memory:
                # memory がまだない場合（初期ターン）は古いメッセージを削除するだけ
                self.remove_temp_messages_from_dialogue(keep_num=2)
                break
            temp_message = None
            if self.dialogue[-1]['role'] == 'user':
                temp_message = self.dialogue[-1]
            self.remove_temp_messages_from_dialogue(keep_num=2)
            self.add_user_message_to_dialogue(
                'The former chat history can be summarized as: \n' + self.memory
            )
            if temp_message is not None:
                self.dialogue.append(temp_message)
            # 圧縮後もトークン数が変わらない場合（現ターン入力だけで上限超過）は
            # これ以上削減できないのでループを抜ける
            if num_tokens_from_messages(self.dialogue, self.model) >= limit:
                fc_logger.warning(
                    f"Token limit ({limit}) still exceeded after compression "
                    f"({num_tokens_from_messages(self.dialogue, self.model)} tokens). "
                    "Proceeding anyway."
                )
                break

    # ------------------------------------------------------------------
    # コマンド生成
    # ------------------------------------------------------------------
    def generate_command(self, prompt: str):
        full_prompt = prompt + "\n" + self.prompt_handler.insist_json()

        print("\n===== PROMPT TO QWEN =====")
        print(full_prompt)

        self.add_user_message_to_dialogue(full_prompt)
        self.restrict_dialogue()

        response = self.query_llm()

        if response is None:
            return None

        # GPT版の memory.save_context() 相当: 毎ターン memory をインクリメンタルに更新
        try:
            assistant_content = response['choices'][0]['message']['content']
            self._update_memory(full_prompt, assistant_content)
        except Exception as e:
            fc_logger.error(f'Memory update failed: {e}')

        return response

    # ------------------------------------------------------------------
    # JSONパース（GPT workerと同じ } 自動補完ロジック）
    # ------------------------------------------------------------------
    def parse_response(self, response):
        try:
            content = response['choices'][0]['message']['content']

            print("\n===== QWEN OUTPUT =====")
            print(content)

            start = content.find('{')
            if start == -1:
                raise ValueError("No JSON object found in response")

            depth = 0
            end = None
            for i, c in enumerate(content[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end is None:
                rlack = content[start:].count('{') - content[start:].count('}')
                json_str = content[start:] + '}' * max(0, rlack)
            else:
                json_str = content[start:end]

            return json.loads(json_str)

        except Exception as e:
            fc_logger.error(f'JSON parse error: {e}')
            return None

    # ------------------------------------------------------------------
    # コマンド実行（GPT workerと同じ構造）
    # ------------------------------------------------------------------
    def process_command(self, response, obs_input_prompt, current_avail_actions):
        try:
            command_json = self.parse_response(response)
            # Qwen sometimes nests "command" inside "thoughts" instead of top-level
            if 'command' in command_json:
                command_block = command_json['command']
            elif 'thoughts' in command_json and 'command' in command_json['thoughts']:
                fc_logger.warning('command found nested inside thoughts — extracting')
                command_block = command_json['thoughts']['command']
            else:
                raise KeyError('command')
            command_name  = command_block['name']
            command_input = command_block['input']
        except Exception as e:
            fc_logger.error(f'\nRESPONSE:{response}\nCommand json parsing error: {e}')
            print('Not in given json format, retrying...')
            return None, self.prompt_handler.insist_json()

        if command_name not in self.command_handlers:
            return None, self.prompt_handler.insist_available_commands(
                list(self.command_handlers.keys())
            )

        try:
            return self.command_handlers[command_name](
                command_input, obs_input_prompt, current_avail_actions
            )
        except Exception as e:
            fc_logger.error(f'Command handler "{command_name}" error: {e}')
            print(f'Command handler "{command_name}" error: {e}, retrying...')
            return None, self.prompt_handler.insist_json()
