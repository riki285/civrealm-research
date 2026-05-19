import json
import requests

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler
from .base_worker import BaseWorker


class QwenWorker(BaseWorker):
    """
    Worker for local Qwen (vLLM) inference.
    Follows the same structure as AzureGPTWorker.
    """

    def __init__(self,
                 model: str = "Lorbus/Qwen3.6-27B-int4-AutoRound",
                 api_url: str = "http://localhost:8001/v1/chat/completions",
                 prompt_prefix: str = "civ_prompts_qwen",
                 temperature: float = 0.3,
                 max_tokens: int = 512,
                 **kwargs):
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_prefix = prompt_prefix
        super().__init__(model, **kwargs)

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(prompt_prefix=self.prompt_prefix)
        self.add_user_message_to_dialogue(self.prompt_handler.instruction_prompt())
        self.add_user_message_to_dialogue(self.prompt_handler.task_prompt())

    def init_llm(self):
        # dialogueをリセットしない（init_promptsで追加したinstruction/task promptを保持）
        self.memory = []

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
                    "chat_template_kwargs": {"enable_thinking": False},#thinking modeオフ
                },
                timeout=120
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

        self.memory.append({"user": prompt, "assistant": str(response)})
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

            # 深さカウンタで正確な閉じ位置を探す（余分な } にも対応）
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
                # 閉じ括弧が足りない場合は補完
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
            command_name  = command_json['command']['name']
            command_input = command_json['command']['input']
        except Exception as e:
            fc_logger.error(f'\nRESPONSE:{response}\nCommand json parsing error: {e}')
            print('Not in given json format, retrying...')
            return None, self.prompt_handler.insist_json()

        if command_name not in self.command_handlers:
            return None, self.prompt_handler.insist_available_commands(
                list(self.command_handlers.keys())
            )

        return self.command_handlers[command_name](
            command_input, obs_input_prompt, current_avail_actions
        )
