from .qwen_mastaba_agent import QwenMastabaAgent
from .workers.rule_based_worker import RuleBasedWorker


class QwenMastabaRuleAgent(QwenMastabaAgent):
    """
    advisor は LLM（Qwen）、controller はルールベースで動作するエージェント。
    QwenMastabaAgent の _create_controller_worker() のみをオーバーライドする。
    """

    def _create_controller_worker(self, entity_type, entity_id):
        return RuleBasedWorker(ctrl_type=entity_type, actor_id=entity_id)
