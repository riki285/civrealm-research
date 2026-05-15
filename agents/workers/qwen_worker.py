import os
import json
import random
import requests

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler
from .base_worker import BaseWorker


class QwenWorker(BaseWorker):
    """
    Qwen (vLLM local server) version of AzureGPTWorker
    """

    def __init__(self,
                 model: str = "Qwen/Qwen2.5-7B-Instruct",
                 api_url: str = "http://localhost:8001/v1/chat/completions",
                 prompt_prefix: str = "civ_prompts",
                 temperature: float = 0.3,
                 max_tokens: int = 128,
                 **kwargs):

        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_prefix = prompt_prefix

        super().__init__(model, **kwargs)

    # -----------------------------
    # 初期化
    # -----------------------------
    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(
            prompt_prefix=self.prompt_prefix)
        self._load_instruction_prompt()
        self._load_task_prompt()

    def init_llm(self):
        # Azureのような初期化は不要
        self.dialogue = []
        self.memory = []  # 軽量メモリ

    def init_index(self):
        # Pineconeは使わない（軽量化）
        self.index = None

    # -----------------------------
    # プロンプト
    # -----------------------------
    def _load_instruction_prompt(self):
        instruction_prompt = self.prompt_handler.instruction_prompt()
        self.add_user_message_to_dialogue(instruction_prompt)

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)

    # -----------------------------
    # コマンド登録
    # -----------------------------
    def register_all_commands(self):
        self.register_command('finalDecision',
                              self.handle_command_final_decision)
        self.register_command('suggestion',
                              self.handle_command_suggestion)

    # -----------------------------
    # コマンド処理
    # -----------------------------
    def handle_command_suggestion(self, command_input,
                                  obs_input_prompt,
                                  current_avail_actions):
        return command_input["suggestion"], ''

    def handle_command_final_decision(self, command_input,
                                      obs_input_prompt,
                                      current_avail_actions):

        exec_action = command_input['action']
        lower_avail_actions = [x.lower() for x in current_avail_actions]

        if exec_action.lower() not in lower_avail_actions:
            fc_logger.error(
                f'Invalid action "{exec_action}", retrying...')
            return None, self.prompt_handler.insist_avail_action()

        return exec_action, ''

    # -----------------------------
    # Qwen呼び出し（核心）
    # -----------------------------
    def query_llm(self):
        try:
            #print("Sending request to Qwen")

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": self.dialogue,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,

                    "guided_json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string"
                            }
                        },
                        "required": ["action"]
                    }
                },timeout=30
            )

            print("Status:", response.status_code)

            if response.status_code != 200:
                print("Error response:", response.text)
                return None

            return response.json()

        except Exception as e:
            print("Exception:", e)
            return None

    # -----------------------------
    # コマンド生成
    # -----------------------------
    def generate_command(self, prompt: str):
        full_prompt = prompt + self.prompt_handler.insist_json()

        print("\n===== PROMPT TO QWEN =====")
        print(full_prompt)

        self.add_user_message_to_dialogue(full_prompt)

        self.restrict_dialogue()

        response = self.query_llm()

        if response is None:
            return None

        self.memory.append({
            "user": prompt,
            "assistant": str(response)
        })

        return response

    # -----------------------------
    # JSONパース
    # -----------------------------
    def parse_response(self, response):
        try:
            content = response['choices'][0]['message']['content']
            
            print("\n===== QWEN OUTPUT =====")
            print(response['choices'][0]['message']['content'])

            start = content.find('{')
            end = content.rfind('}') + 1
            content = content[start:end]

            return json.loads(content)

        except Exception as e:
            fc_logger.error(f"JSON parse error: {e}")
            return None

    # -----------------------------
    # コマンド実行
    # -----------------------------
    def process_command(self, response,
                        obs_input_prompt,
                        current_avail_actions):

        command_json = self.parse_response(response)

        if command_json is None:
            return None, self.prompt_handler.insist_json()

        if 'command' in command_json:
            command_name = command_json['command']['name']
            command_input = command_json['command']['input']
        else:
            # Qwen simple JSON format
            command_name = "finalDecision"
            command_input = command_json

        if command_name not in self.command_handlers:
            return None, self.prompt_handler.insist_available_commands(
                list(self.command_handlers.keys())
            )

        return self.command_handlers[command_name](
            command_input,
            obs_input_prompt,
            current_avail_actions
        )