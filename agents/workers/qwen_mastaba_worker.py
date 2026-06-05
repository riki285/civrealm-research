from agents.prompt_handlers.base_prompt_handler import BasePromptHandler
from .qwen_worker import QwenWorker


class QwenMastabaWorker(QwenWorker):
    """
    QwenWorker に advisor / controller の役割を追加したワーカー。
    - advisor : ターン開始時に全体戦略を提案する（suggestion コマンドを返す）
    - controller: 個別エンティティの行動を決定する（finalDecision コマンドを返す）
    """

    def __init__(self, role: str = "controller", **kwargs):
        self.role = role
        super().__init__(**kwargs)

    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(prompt_prefix=self.prompt_prefix)
        if self.role == "advisor":
            self.add_user_message_to_dialogue(
                self.prompt_handler.advisor_instruction_nomap()
            )
            self.add_user_message_to_dialogue(
                self.prompt_handler.advisor_task()
            )
        else:
            self.add_user_message_to_dialogue(
                self.prompt_handler.instruction_prompt()
            )
            self.add_user_message_to_dialogue(
                self.prompt_handler.task_prompt()
            )
